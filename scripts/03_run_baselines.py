"""Compare time interpolation and fixed graph smoothing on hidden measurements."""

import argparse
from collections import defaultdict
from importlib import import_module

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

graphs = import_module("02_build_graphs")
audit, plt = graphs.audit, graphs.plt
RESULTS = graphs.prepare.OUTPUTS / "03_baselines"
MASK_LABELS = {"global_alternating": "Alternating whole frames", "local_gap": "Local spatial gap"}
METHODS = {"linear": "Linear", "pchip": "PCHIP", "pchip_graph": "PCHIP + graph"}
COLORS = {"linear": "#2088a6", "pchip": "#c55b58", "pchip_graph": "#926cb2"}


def observation_masks(coords_mm: np.ndarray, n_times: int) -> dict[str, np.ndarray]:
    n_nodes = len(coords_mm)
    global_hidden = np.zeros((n_nodes, n_times), dtype=bool)
    global_hidden[:, 1:-1:2] = True

    # Hide the same interval on a compact set of spatially nearby nodes.
    local_hidden = np.zeros_like(global_hidden)
    center = np.median(coords_mm, axis=0)
    distance = np.linalg.norm(coords_mm - center, axis=1)
    nodes = np.argsort(distance, kind="stable")[:max(1, n_nodes // 4)]
    width = max(1, (n_times - 2) // 3)
    start = max(1, n_times // 3 - width // 2)
    local_hidden[nodes, start:start + width] = True
    return {"global_alternating": global_hidden, "local_gap": local_hidden}


def time_interpolation(times: np.ndarray, observed: np.ndarray, hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    linear, pchip = observed.copy(), observed.copy()
    for node, gap in enumerate(hidden):
        if not gap.any():
            continue
        t_visible = times[~gap]
        y_visible = observed[node, ~gap]
        linear[node, gap] = np.interp(times[gap], t_visible, y_visible)
        pchip[node, gap] = PchipInterpolator(t_visible, y_visible, extrapolate=False)(times[gap])
    return linear, pchip


def graph_smoothing(prior: np.ndarray, observed: np.ndarray, hidden: np.ndarray,
                    edges: np.ndarray, weight: float) -> np.ndarray:
    # Per time point, minimize squared deviation from the PCHIP prior plus
    # weight * sum((x[i] - x[j])**2 over edges), fixing all observed x values.
    n_nodes = len(prior)
    src, dst = edges.T
    adjacency = coo_matrix((np.ones(2 * len(edges)), (np.r_[src, dst], np.r_[dst, src])),
                           shape=(n_nodes, n_nodes)).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    system = (diags(1 + weight * degree, format="csr") - weight * adjacency).tocsr()
    result = prior.copy()
    for t, gap in enumerate(hidden.T):
        missing_nodes = np.flatnonzero(gap)
        if not missing_nodes.size:
            continue
        visible_nodes = np.flatnonzero(~gap)
        rhs = prior[missing_nodes, t] + weight * (adjacency[missing_nodes][:, visible_nodes] @ observed[visible_nodes, t])
        result[missing_nodes, t] = spsolve(system[missing_nodes][:, missing_nodes].tocsc(), rhs)
    return result


def save_examples(sample: str, graph, cases: dict) -> dict:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), width_ratios=(1, 1.55), layout="constrained")
    times = graph["times_s"]
    labels = graph["labels"]
    reference = graph["reference"]
    limits = np.percentile(reference[labels > 0], [1, 99])
    examples = {}
    for (name, (hidden, predictions)), (map_ax, curve_ax) in zip(cases.items(), axes):
        hidden_frames = np.flatnonzero(hidden.any(axis=0))
        t = hidden_frames[len(hidden_frames) // 2]
        candidates = np.flatnonzero(hidden[:, t])
        node = candidates[len(candidates) // 2]  # Choose before looking at prediction errors.
        spatial_mask = np.isin(labels, candidates + 1)
        map_ax.imshow(reference, cmap="gray", vmin=limits[0], vmax=limits[1])
        map_ax.imshow(np.ma.masked_where(~spatial_mask, spatial_mask), cmap="autumn", vmin=0, vmax=1, alpha=0.45)
        r, c = graph["centroids_rc"][node]
        map_ax.scatter(c, r, s=34, c="white", edgecolors="black", linewidths=0.6)
        map_ax.set_title(f"{MASK_LABELS[name]}\nHidden regions at TT {times[t]:g} s")
        map_ax.axis("off")

        truth = graph["signals_pct"][node]
        gap = hidden[node]
        examples[name] = dict(node=int(node), nodes=len(candidates), time=times[t], steps=int(gap.sum()),
                              first=times[gap][0], last=times[gap][-1])
        visible = ~gap
        curve_ax.plot(times, truth, color="#303840", linestyle="--", lw=1, label="Measured reference")
        curve_ax.scatter(times[visible], truth[visible], s=15, color="#303840", label="Given to all methods")
        for method, prediction in predictions.items():
            curve_ax.scatter(times[gap], prediction[node, gap], s=28, facecolors="none",
                             edgecolors=COLORS[method], label=METHODS[method])
        curve_ax.set(title=f"Node {node} | Open markers = estimates", xlabel="TT (s)",
                     ylabel="Change vs. reference (%)")
        curve_ax.grid(alpha=0.18)
    axes[0, 1].legend(fontsize=8, frameon=False, ncol=2)
    fig.suptitle(sample)
    fig.savefig(RESULTS / "baseline_examples.png", dpi=150)
    plt.close(fig)
    return examples


def summarize_metrics(metrics: list[dict]) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for row in metrics:
        groups[(row["patient"], row["mask"], row["method"])].append(row)
    patient_rows = []
    for (patient, mask, method), visits in sorted(groups.items()):
        patient_rows.append(dict(patient=patient, mask=mask, method=method, visits=len(visits),
                                 mae_pp=float(np.mean([v["mae_pp"] for v in visits])),
                                 rmse_pp=float(np.mean([v["rmse_pp"] for v in visits]))))
    summary = []
    for mask in MASK_LABELS:
        for method in METHODS:
            cohort = [row for row in patient_rows if row["mask"] == mask and row["method"] == method]
            summary.append(dict(mask=mask, method=method, patients=len(cohort),
                                mae_pp=float(np.mean([r["mae_pp"] for r in cohort])),
                                rmse_pp=float(np.mean([r["rmse_pp"] for r in cohort]))))
    return patient_rows, summary


def save_comparison(summary: list[dict], patients: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), sharey=True, layout="constrained")
    for ax, mask in zip(axes, MASK_LABELS):
        for x, method in enumerate(METHODS):
            mean = next(row["mae_pp"] for row in summary if row["mask"] == mask and row["method"] == method)
            points = [row["mae_pp"] for row in patients if row["mask"] == mask and row["method"] == method]
            ax.bar(x, mean, color=COLORS[method], width=0.6, alpha=0.65)
            ax.scatter(x + np.linspace(-0.13, 0.13, len(points)), points, s=16, color="#263849", zorder=3)
        ax.set_xticks(range(len(METHODS)), list(METHODS.values()))
        ax.set_title(MASK_LABELS[mask])
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Hidden-point MAE (percentage points)")
    fig.suptitle("Bars: mean over patients. Dots: individual patients.")
    fig.savefig(RESULTS / "baseline_comparison.png", dpi=150)
    plt.close(fig)


def describe_results(summary: list[dict], patients: list[dict]) -> str:
    notes = []
    labels = {"global_alternating": "整帧隔点隐藏", "local_gap": "局部连续缺失"}
    for mask, label in labels.items():
        means = {row["method"]: row["mae_pp"] for row in summary if row["mask"] == mask}
        winner = min(means, key=means.__getitem__)
        per_patient = defaultdict(dict)
        for row in patients:
            if row["mask"] == mask:
                per_patient[row["patient"]][row["method"]] = row["mae_pp"]
        worse = sum(scores["pchip_graph"] > scores["pchip"] for scores in per_patient.values())
        notes.append(
            f'<li><strong>{label}</strong>。{METHODS[winner]} 的平均 MAE 最低，为 {means[winner]:.2f} 个百分点。'
            f'加图平滑后，PCHIP 的 MAE 从 {means["pchip"]:.2f} 变为 {means["pchip_graph"]:.2f} 个百分点，'
            f'{worse}/{len(per_patient)} 位患者的 MAE 上升。</li>'
        )
    return "<ul>" + "".join(notes) + "</ul>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    source = graphs.GRAPH_DIR / "manifest.csv"
    if not source.is_file():
        parser.error("Run 02_build_graphs.py first.")
    rows = audit.read_csv(source)
    if not rows:
        parser.error("The graph manifest is empty.")
    mask_dir = RESULTS / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    graph_weight = 0.25
    metrics = []
    for index, row in enumerate(rows):
        sample = row["sample"]
        print(f"Scoring {sample}", flush=True)
        with np.load(graphs.GRAPH_DIR / row["file"], allow_pickle=False) as graph:
            truth, times = graph["signals_pct"], graph["times_s"]
            masks = observation_masks(graph["centroids_mm"], len(times))
            cases = {}
            for mask, hidden in masks.items():
                observed = np.where(hidden, np.nan, truth).astype(float)
                linear, pchip = time_interpolation(times, observed, hidden)
                predictions = dict(linear=linear, pchip=pchip,
                                   pchip_graph=graph_smoothing(pchip, observed, hidden, graph["edges"], graph_weight))
                for method, prediction in predictions.items():
                    error = prediction[hidden] - truth[hidden]
                    if not np.isfinite(error).all():
                        raise ValueError(f"Non-finite reconstruction for {sample}, {mask}, {method}.")
                    metrics.append(dict(sample=sample, patient=row["patient"], visit=row["visit"],
                                        mask=mask, method=method, hidden_points=error.size,
                                        hidden_fraction=error.size / truth.size,
                                        mae_pp=float(np.abs(error).mean()), rmse_pp=float(np.sqrt(np.square(error).mean()))))
                if index == 0:
                    cases[mask] = (hidden, predictions)
            np.savez_compressed(mask_dir / f"{sample}.npz", times_s=times, **masks)
            if index == 0:
                examples = save_examples(sample, graph, cases)

    patients, summary = summarize_metrics(metrics)
    audit.write_csv(RESULTS / "metrics_by_visit.csv", metrics)
    audit.write_csv(RESULTS / "metrics_by_patient.csv", patients)
    audit.write_csv(RESULTS / "summary.csv", summary)
    save_comparison(summary, patients)
    table = audit.html_table(["缺失方式", "方法", "患者数", "平均 MAE (百分点)", "平均 RMSE (百分点)"], [
        (MASK_LABELS[r["mask"]], METHODS[r["method"]], r["patients"], f"{r['mae_pp']:.3f}", f"{r['rmse_pp']:.3f}")
        for r in summary
    ])
    example = rows[0]["sample"]
    whole, local = examples["global_alternating"], examples["local_gap"]
    result_notes = describe_results(summary, patients)
    graphs.prepare.write_report(RESULTS, "03 · 藏起一段曲线，看哪种方法补得准", f"""
<section><h2>03 做了什么</h2>
<p><a href="../02_graphs/index.html">02</a> 给了我们每个区域的实测信号曲线和邻居名单。03 做了一次「遮住再补」的实验。我们先故意藏起一部分实测值，只把剩下的值给三种方法。补完后揭开原来的值，比较各方法猜得有多准。这些方法是以后评估 GNN 的参照，也叫 baseline。</p>
<p>首尾观测保留，所以方法能同时用缺口前后的测量。这里做的是同一次检查内的曲线补全。</p></section>
<section><h2>第一张图怎么读</h2>
<figure><img src="baseline_examples.png" alt="上排整帧缺失，下排局部缺失。左侧显示缺失区域，右侧显示选定节点的观测与估计"></figure>
<ol>
<li><strong>上排是整帧隔点隐藏。</strong>在 {example}，图左所示 TT {whole["time"]:g} 秒的黄色覆盖全部 {whole["nodes"]} 个节点。当某一整帧被藏起来，空间邻居在同一时刻也缺失。白点所在的节点 {whole["node"]} 整条曲线里藏了 {whole["steps"]} 个时间点。</li>
<li><strong>下排是局部区域连续隐藏。</strong>黄色仅覆盖一组空间靠近的节点。在这例里，共有 {local["nodes"]} 个节点在 TT {local["first"]:g} 到 {local["last"]:g} 秒之间连续缺失 {local["steps"]} 帧。图左截取 TT {local["time"]:g} 秒，白点对应右图的节点 {local["node"]}。黄色区域外的节点在这些时刻仍有测量。左图灰底用来定位，曲线里的时间看右图横轴。</li>
<li><strong>在任一张右图里选一个空心圈。</strong>黑色实心点是事先给方法看的值。黑色虚线把实测值连起来供我们核对，包括被藏的那些时刻。同一个横坐标上，蓝、红、紫空心圈分别是三种估计。空心圈在竖直方向离黑色虚线越近，该点误差越小。局部长缺口里，尤其值得看空心圈能否跟上虚线的峰和谷。</li>
</ol></section>
<section><h2>三种方法用了哪些信息</h2>
<ul>
<li><strong>蓝色 · Linear（线性插值）</strong>。对每个节点，只看该节点已有的测量，把缺口两侧最近的点用直线连起来。</li>
<li><strong>红色 · PCHIP（保形三次插值）</strong>。仍使用同一节点的观测，用弯曲的线段补齐缺口，尽量保持已有点的变化形态。</li>
<li><strong>紫色 · PCHIP + graph（固定图平滑）</strong>。先做 PCHIP，再在缺失时刻拉近相邻节点的估计。已有观测保持原样。整帧缺失时，参与平滑的是这一帧各节点的估计；局部缺失时，还能用到缺口外节点在同一时刻的实测值。本轮平滑权重固定为 {graph_weight:g}。</li>
</ul></section>
<section><h2>第二张图和结果表</h2>
{table}
<figure><img src="baseline_comparison.png" alt="两类缺失下三种方法的平均绝对误差"><figcaption>左面板是整帧隔点隐藏，右面板是局部连续缺失。每个黑点代表一位患者，每根柱子是所有患者的等权平均。柱越矮，补得越准。</figcaption></figure>
<p>图用 MAE（平均绝对误差）。例如实测相对变化为 +40%，估计为 +35%，这点的绝对误差就是 5 个百分点。先对每次检查的隐藏点计算误差，再对同一患者的访视取均值，最后对患者等权平均。表中的 RMSE 同样用百分点表示，对较大的误差更敏感。</p>
{result_notes}
<p>这里评估的是两种人工遮挡方式和这一种固定平滑规则。空间相邻的区域可能具有不同的时间曲线，平滑会拉近它们，这是误差变大的一种可能解释。要评估 GNN，后续可以用相同缺失掩码与线性插值比较，并按患者留出测试数据。</p>
<p><a href="metrics_by_visit.csv">访视明细</a> · <a href="metrics_by_patient.csv">逐患者指标</a> · <a href="summary.csv">汇总表</a></p></section>""")
    print(f"Report: {RESULTS / 'index.html'}")


if __name__ == "__main__":
    main()
