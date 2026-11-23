"""Score held-out patients on the 03 masks and explain example GNN corrections."""

import argparse
import json
from collections import Counter, defaultdict
from html import escape
from importlib import import_module

import numpy as np
import torch

training = import_module("04_train_gnn")
audit, plt = training.audit, training.plt
MASK_LABELS = training.baseline.MASK_LABELS
MASK_ZH = {"global_alternating": "整帧隔点隐藏", "local_gap": "局部连续缺失"}
EVALUATION = training.graphs.prepare.OUTPUTS / "05_evaluation"
LINEAR_COLOR = "#2088a6"
GNN_COLOR = "#8259a4"


def check_splits(record: dict, patients: set[str]) -> None:
    tested = Counter()
    for split in record["folds"]:
        roles = [split["train"], split["validation"], split["test"]]
        members = [patient for group in roles for patient in group]
        if not all(roles) or set(members) != patients or len(members) != len(patients):
            raise ValueError(f"Patient roles overlap or are incomplete in fold {split['fold']}.")
        tested.update(split["test"])
    if tested != Counter(patients):
        raise ValueError("Every patient must be tested exactly once.")


def scores(prediction: np.ndarray, truth: np.ndarray, hidden: np.ndarray) -> tuple[float, float]:
    error = prediction[hidden] - truth[hidden]
    if not np.isfinite(error).all():
        raise ValueError("A prediction contains non-finite values.")
    return float(np.abs(error).mean()), float(np.sqrt(np.square(error).mean()))


def patient_results(visits: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in visits:
        grouped[row["patient"], row["mask"]].append(row)
    rows = []
    for (patient, mask), cases in sorted(grouped.items()):
        averaged = {key: float(np.mean([c[key] for c in cases]))
                    for key in ("linear_mae_pp", "gnn_mae_pp", "linear_rmse_pp", "gnn_rmse_pp")}
        rows.append(dict(patient=patient, fold=cases[0]["fold"], mask=mask, visits=len(cases),
                         **averaged, mae_delta_pp=averaged["gnn_mae_pp"] - averaged["linear_mae_pp"]))
    return rows


def save_comparison(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.5), layout="constrained", sharex=True, sharey=True)
    limit = 1.08 * max(1.0, max(max(row["linear_mae_pp"], row["gnn_mae_pp"]) for row in rows))
    for ax, (mask, title) in zip(axes, MASK_LABELS.items()):
        subset = [row for row in rows if row["mask"] == mask]
        ax.plot([0, limit], [0, limit], linestyle="--", color="#405266", label="Equal error")
        ax.scatter([r["linear_mae_pp"] for r in subset], [r["gnn_mae_pp"] for r in subset],
                   s=40, color="#749ac6", edgecolors="#222d5a", zorder=3)
        ax.set(xlim=(0, limit), ylim=(0, limit), aspect="equal", title=title, xlabel="Linear MAE (pp)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("GNN MAE (pp)")
    axes[0].legend(frameon=False)
    fig.savefig(EVALUATION / "paired_patient_mae.png", dpi=150)
    plt.close(fig)


def explain_case(visit, mask: str, case, prediction, weights, contributions, fold: int) -> tuple[dict, list[dict]]:
    hidden = case.hidden.numpy()
    src, dst = visit.src.numpy(), visit.dst.numpy()
    times = visit.times
    selected_times = np.flatnonzero(hidden.any(axis=0))
    t = selected_times[len(selected_times) // 2]
    degree = np.bincount(dst, minlength=len(hidden))
    observed_neighbors = np.bincount(dst, weights=(~hidden[src, t]), minlength=len(hidden))
    candidates = np.flatnonzero(hidden[:, t] & (degree > 0))
    if not candidates.size:
        candidates = np.flatnonzero(hidden[:, t])
    # Pick using mask and graph only. Prefer visible neighbors in the local example.
    counts = observed_neighbors[candidates]
    candidates = candidates[counts == counts.max()]
    node = int(candidates[len(candidates) // 2])
    incoming = np.flatnonzero(dst == node)
    neighbors = src[incoming]
    ranking = incoming[np.argsort(-np.abs(contributions[incoming, t]), kind="stable")]
    details = [dict(sample=visit.sample, fold=fold, mask=mask, node=node, tt_s=times[t],
                    neighbor=int(src[edge]), observed=bool(not hidden[src[edge], t]),
                    neighbor_value_pct=float(case.linear[src[edge], t]), weight=float(weights[edge, t]),
                    contribution_pp=float(contributions[edge, t])) for edge in ranking]
    estimate = float(prediction[node, t])
    total = sum(row["contribution_pp"] for row in details)
    if not np.isclose(estimate, case.linear[node, t] + total, atol=1e-4, rtol=1e-6):
        raise ValueError("Neighborhood explanation does not add up to the prediction.")

    with np.load(visit.graph_path, allow_pickle=False) as graph:
        labels, reference, coords = graph["labels"], graph["reference"], graph["centroids_rc"][:, ::-1]
    vmin, vmax = np.percentile(reference[labels > 0], [1, 99])
    fig, (map_ax, curve_ax, bar_ax) = plt.subplots(1, 3, figsize=(12.3, 4.1),
                                                  width_ratios=(1, 1.65, 1), layout="constrained")
    map_ax.imshow(reference, cmap="gray", vmin=vmin, vmax=vmax)
    selected_region = labels == node + 1
    map_ax.imshow(np.ma.masked_where(~selected_region, selected_region), cmap="autumn", vmin=0, vmax=1, alpha=0.55)
    for neighbor in neighbors:
        color = "#49d4e8" if not hidden[neighbor, t] else "#c597df"
        map_ax.plot([coords[node, 0], coords[neighbor, 0]], [coords[node, 1], coords[neighbor, 1]],
                    color=color, linewidth=1)
        map_ax.scatter(*coords[neighbor], color=color, s=21, edgecolor="#263849", linewidth=0.5)
    map_ax.scatter(*coords[node], color="#ffe06a", edgecolor="#263849", marker="*", s=112, zorder=4)
    selected_coords = coords[np.r_[node, neighbors]]
    if len(selected_coords) > 1:
        x, y = selected_coords.T
        map_ax.set_xlim(max(-0.5, x.min() - 30), min(reference.shape[1] - 0.5, x.max() + 30))
        map_ax.set_ylim(min(reference.shape[0] - 0.5, y.max() + 30), max(-0.5, y.min() - 30))
    map_ax.set_title(f"Node {node} and direct neighbors")
    map_ax.axis("off")

    gap = hidden[node]
    curve_ax.plot(times, case.truth[node], color="#303840", ls="--", lw=1, label="Measured reference")
    curve_ax.scatter(times[~gap], case.truth[node, ~gap], color="#303840", s=14, label="Observed")
    curve_ax.scatter(times[gap], case.linear[node, gap], edgecolors=LINEAR_COLOR, facecolors="none",
                     s=29, label="Linear")
    curve_ax.scatter(times[gap], prediction[node, gap], edgecolors=GNN_COLOR, facecolors="none",
                     marker="s", s=24, label="GNN")
    curve_ax.axvline(times[t], color="#b16a35", ls=":", lw=1.2)
    curve_ax.set(xlabel="TT (s)", ylabel="Change vs. reference (%)", title=f"Target curve at TT {times[t]:g} s")
    curve_ax.grid(alpha=0.2)
    curve_ax.legend(frameon=False, fontsize=7)

    if details:
        labels_y = [str(row["neighbor"]) for row in details]
        values = [row["contribution_pp"] for row in details]
        bar_ax.barh(labels_y, values, color=["#2088a6" if v >= 0 else "#cd7f6d" for v in values])
        bar_ax.invert_yaxis()
    else:
        bar_ax.text(0.5, 0.5, "No neighbors", ha="center", transform=bar_ax.transAxes)
    bar_ax.axvline(0, color="#405266", linewidth=0.8)
    bar_ax.set(title="Signed neighbor contributions", xlabel="Adjustment (pp)", ylabel="Neighbor node")
    fig.suptitle(f"Test sample {visit.sample} | Fold {fold} | {MASK_LABELS[mask]}")
    fig.savefig(EVALUATION / f"neighbors_{mask}.png", dpi=145)
    plt.close(fig)
    return dict(sample=visit.sample, node=node, time=times[t], truth=float(case.truth[node, t]),
                linear=float(case.linear[node, t]), correction=total, gnn=estimate), details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive.")
    device, device_label = training.configure_device(parser, args)
    record_path = training.TRAINING / "run.json"
    if not record_path.is_file():
        parser.error("Run 04_train_gnn.py to completion first.")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("version") != 1:
        parser.error("Unsupported 04 run format.")
    try:
        samples, fingerprint = training.load_data()
        check_splits(record, {visit.patient for visit in samples})
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    if record["inputs"] != fingerprint:
        parser.error("The graph files or the 03 masks changed after training. Rerun 04 before evaluating.")
    recorded_linear = {(row["sample"], row["mask"]): row for row in
                       audit.read_csv(training.baseline.RESULTS / "metrics_by_visit.csv") if row["method"] == "linear"}
    expected = {(visit.sample, mask) for visit in samples for mask in MASK_LABELS}
    if set(recorded_linear) != expected:
        parser.error("The 03 linear metrics do not cover the current graph and mask cases. Rerun 03 and 04.")
    for visit in samples:
        for mask, case in visit.cases.items():
            hidden = case.hidden.numpy()
            row = recorded_linear[visit.sample, mask]
            measured = scores(case.linear, case.truth, hidden)
            if (row["patient"] != visit.patient or row["visit"] != visit.visit or
                    np.count_nonzero(hidden) != int(row["hidden_points"]) or
                    not np.allclose(measured, [float(row["mae_pp"]), float(row["rmse_pp"])], atol=1e-8, rtol=1e-8)):
                parser.error(f"Linear scores do not match 03 for {visit.sample}, {mask}. Rerun 03 and 04.")

    run_dir = training.TRAINING / "runs" / record["run_id"]
    EVALUATION.mkdir(parents=True, exist_ok=True)
    example_sample = samples[0].sample
    visits, examples, detail_rows = [], {}, []
    for split in record["folds"]:
        fold = split["fold"]
        model = training.NeighborGNN(record["parameters"]["width"])
        model.load_state_dict(torch.load(run_dir / split["checkpoint"], map_location="cpu", weights_only=True))
        model.to(device).eval()
        with torch.inference_mode():
            for visit in samples:
                if visit.patient not in split["test"]:
                    continue
                print(f"Fold {fold} test | {visit.sample}", flush=True)
                device_visit = training.visit_to(visit, device)
                for mask, case in visit.cases.items():
                    prediction, weights, contributions = model(device_visit, device_visit.cases[mask], explain=True)
                    prediction = prediction.cpu().numpy() * training.SIGNAL_SCALE
                    hidden = case.hidden.numpy()
                    linear_mae, linear_rmse = scores(case.linear, case.truth, hidden)
                    gnn_mae, gnn_rmse = scores(prediction, case.truth, hidden)
                    visits.append(dict(sample=visit.sample, patient=visit.patient, visit=visit.visit, fold=fold,
                                       mask=mask, hidden_points=int(hidden.sum()), linear_mae_pp=linear_mae,
                                       gnn_mae_pp=gnn_mae, linear_rmse_pp=linear_rmse, gnn_rmse_pp=gnn_rmse,
                                       mae_delta_pp=gnn_mae - linear_mae))
                    if visit.sample == example_sample:
                        examples[mask], detail = explain_case(visit, mask, case, prediction, weights.cpu().numpy(),
                                                               contributions.cpu().numpy() * training.SIGNAL_SCALE, fold)
                        detail_rows.extend(detail)

    patients = patient_results(visits)
    summary = []
    for mask in MASK_LABELS:
        subset = [row for row in patients if row["mask"] == mask]
        averaged = {key: float(np.mean([p[key] for p in subset])) for key in
                    ("linear_mae_pp", "gnn_mae_pp", "linear_rmse_pp", "gnn_rmse_pp", "mae_delta_pp")}
        summary.append(dict(mask=mask, patients=len(subset), improved=sum(p["mae_delta_pp"] < 0 for p in subset),
                            **averaged))
    audit.write_csv(EVALUATION / "metrics_by_visit.csv", visits)
    audit.write_csv(EVALUATION / "metrics_by_patient.csv", patients)
    audit.write_csv(EVALUATION / "summary.csv", summary)
    if detail_rows:
        audit.write_csv(EVALUATION / "neighbor_contributions.csv", detail_rows)
    save_comparison(patients)
    comparison = audit.html_table(["缺失方式", "测试患者数", "线性 MAE", "GNN MAE", "GNN − 线性", "GNN 改善人数"], [
        (MASK_ZH[r["mask"]], r["patients"], f"{r['linear_mae_pp']:.2f}", f"{r['gnn_mae_pp']:.2f}",
         f"{r['mae_delta_pp']:+.2f}", f"{r['improved']}/{r['patients']}") for r in summary
    ])
    illustrations = []
    for mask, example in examples.items():
        details = [r for r in detail_rows if r["mask"] == mask]
        table = audit.html_table(["邻居编号", "该时刻数据", "邻居值 (%)", "权重", "加权贡献 (百分点)"], [
            (r["neighbor"], "实测" if r["observed"] else "线性估计", f"{r['neighbor_value_pct']:+.2f}",
             f"{r['weight']:.3f}", f"{r['contribution_pp']:+.2f}") for r in details
        ]) if details else "<p>这个节点没有直接邻居，修正量为零。</p>"
        illustrations.append(f"""<section><h2>{MASK_ZH[mask]}的单点拆解</h2>
<figure><img src="neighbors_{mask}.png" alt="目标和邻居位置、目标实测与估计曲线、邻居贡献条形图"></figure>
<p>示例 {example['sample']}，节点 {example['node']}，TT {example['time']:g} 秒。线性值 {example['linear']:+.2f}%，
所有邻居的贡献合计 {example['correction']:+.2f} 个百分点，GNN 值 {example['gnn']:+.2f}%。原实测值 {example['truth']:+.2f}% 用来核对。</p>
{table}</section>""")
    training.graphs.prepare.write_report(EVALUATION, "05 · 留出的患者上，邻居帮助补得更准吗？", f"""
<section><h2>先看整体结果</h2>
<p>本轮 GNN 推理设备为 <code>{escape(device_label)}</code>。指标统计和绘图在 CPU 上完成。</p>
<p>每位患者只用从未对该患者训练或选轮次的模型评估。两种方法在完全相同的隐藏点上比较，线性分数逐次访视核对到 03。先计算每次访视误差，再对同一患者的访视平均，最后对患者等权平均。表里的误差单位都是百分点，<strong>GNN − 线性为负表示 GNN 更准</strong>。</p>
{comparison}
<figure><img src="paired_patient_mae.png" alt="按缺失模式分面的逐患者配对误差图">
<figcaption>左图是整帧缺失，右图是局部缺失。每点是一位测试患者，横坐标为线性 MAE，纵坐标为 GNN MAE。点在虚线下方表示这位患者用 GNN 补得更准，虚线上方则表示线性插值更准。</figcaption></figure>
<p><a href="metrics_by_patient.csv">逐患者结果</a> · <a href="metrics_by_visit.csv">逐访视结果</a> · <a href="summary.csv">含 RMSE 的汇总表</a> · <a href="../04_training/index.html">训练与划分</a>。</p></section>
<section><h2>再跟踪一次具体的修正</h2>
<ol><li>从左图的黄星出发，它是要补的节点。相连的青点是这个时刻有实测值的邻居，紫点在这个时刻只有各自的线性估计。</li>
<li>中图只画黄星节点的曲线。黑实点为模型能看到的测量，黑虚线是完整实测曲线。蓝圈和紫方块分别是线性与 GNN 的补值。竖虚线对应右图正在拆解的时刻。</li>
<li>右图每条横柱是一个邻居加权后的实际贡献。向右抬高线性值，向左降低线性值。把这些柱子的有符号数值相加，再加到线性值上，得到紫方块。权重本身只表示这条边占的份额，方向看有符号的贡献。</li></ol>
<p>两张图来自同一个预先选定的测试访视。示例节点按掩码和邻接选择，局部缺失时优先展示仍能接收到实测邻居的节点。</p></section>
{''.join(illustrations)}
<section><p>这些贡献解释的是模型怎样算出该点的修正。整体误差比较检验这套模型在当前数据和缺失设置下的表现。要单独测量正确邻接关系的价值，可以另做遮蔽邻居或打乱连接实验。</p></section>""")
    print(f"Report: {EVALUATION / 'index.html'}")


if __name__ == "__main__":
    main()
