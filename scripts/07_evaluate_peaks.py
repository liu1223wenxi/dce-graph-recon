"""Measure peak-enhancement error after local-gap reconstruction on held-out patients."""

import argparse
import json
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

ablation = import_module("06_ablate_neighbors")
training, audit, plt = ablation.training, ablation.audit, ablation.plt
OUTPUT = training.graphs.prepare.OUTPUTS / "07_peaks"
MASK = "local_gap"
METHODS = ("linear", "time_only", "gnn")
METRICS = tuple(f"{m}_peak_mae_pp" for m in METHODS)
SUBSETS = {"all_affected": "所有受缺失影响的曲线", "peak_affected": "参考峰值所在的帧受缺失影响"}


def roi_rows(path: Path, visits: list) -> dict:
    """Read a manifest of externally verified, image-grid-aligned binary lesion masks."""
    rows = audit.read_csv(path)
    if not rows or not {"sample", "study_uid", "slice_number", "mask_path"} <= rows[0].keys():
        raise ValueError("ROI manifest must include sample,study_uid,slice_number,mask_path.")
    prepared = {row["sample"]: row for row in
                audit.read_csv(training.graphs.prepare.PREPARED_DIR / "manifest.csv")}
    allowed = {visit.sample for visit in visits}
    result = {}
    for row in rows:
        sample = row["sample"]
        if sample not in allowed or sample in result:
            raise ValueError(f"ROI sample is unknown or duplicated: {sample}.")
        expected = prepared[sample]
        if row["study_uid"] != expected["study_uid"] or int(row["slice_number"]) != int(expected["slice_number"]):
            raise ValueError(f"ROI StudyInstanceUID or slice number does not match 01: {sample}.")
        mask_path = path.parent / row["mask_path"]
        if mask_path.suffix.lower() != ".npy":
            raise ValueError(f"Expected a binary 2D .npy mask for {sample}.")
        mask = np.load(mask_path, allow_pickle=False)
        shape = (int(expected["rows"]), int(expected["columns"]))
        if mask.shape != shape or not np.isin(mask, [0, 1]).all() or not mask.any():
            raise ValueError(f"ROI for {sample} must be nonempty, binary and shape {shape}.")
        result[sample] = mask.astype(bool)
    return result


def lesion_curves(visit, case, predictions: dict, roi: np.ndarray):
    with np.load(visit.graph_path, allow_pickle=False) as graph:
        labels = graph["labels"]
    included = roi & (labels > 0)
    coverage = int(included.sum()) / int(roi.sum())
    if coverage < 0.95:
        return None, coverage
    count = np.bincount(labels[included] - 1, minlength=len(case.truth))
    selected = count > 0
    weights = count[selected].astype(float) / count.sum()
    # The same node-level proxy defines both the reference and reconstructed ROI curves.
    truth = (weights @ case.truth[selected])[None, :]
    estimates = {name: (weights @ curve[selected])[None, :] for name, curve in predictions.items()}
    affected_frames = case.hidden.numpy()[selected].any(axis=0)[None, :]
    return (truth, estimates, affected_frames, ["lesion"]), coverage


def peak_metrics(truth: np.ndarray, prediction: dict, hidden: np.ndarray, labels: list) -> list[dict]:
    peaks = truth.max(axis=1)
    # If several measured frames tie for the peak, an unaffected one still gives that peak height.
    peak_frames = np.isclose(truth, peaks[:, None], atol=1e-6, rtol=0)
    peak_affected = np.all(~peak_frames | hidden, axis=1)
    predicted_peaks = {name: value.max(axis=1) for name, value in prediction.items()}
    return [dict(unit=labels[i], affected_frames=int(hidden[i].sum()), peak_affected=bool(peak_affected[i]),
                 reference_peak_pct=float(peaks[i]),
                 **{f"{name}_peak_pct": float(predicted_peaks[name][i]) for name in METHODS},
                 **{f"{name}_peak_error_pp": float(abs(predicted_peaks[name][i] - peaks[i])) for name in METHODS})
            for i in np.flatnonzero(hidden.any(axis=1))]


def save_plot(output: Path, patients: list[dict], example: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), layout="constrained", width_ratios=(1, 1.35))
    ax, curve = axes
    labels = ["All affected", "Reference peak affected"]
    for idx, (subset, title) in enumerate(zip(SUBSETS, labels)):
        group = [p for p in patients if p["analysis"] == subset]
        if not group:
            continue
        for m, metric in zip(METHODS, METRICS):
            position = idx + {"linear": -0.25, "time_only": 0, "gnn": 0.25}[m]
            values = [row[metric] for row in group]
            ax.scatter(np.full(len(values), position), values, color=ablation.COLORS[m], s=13, alpha=0.45)
            ax.scatter(position, np.mean(values), color=ablation.COLORS[m], marker="D", s=50,
                       edgecolor="white", linewidth=0.6)
    for m in METHODS:
        ax.scatter([], [], color=ablation.COLORS[m], label=ablation.MODEL_LABELS[m])
    ax.set(xticks=[0, 1], xticklabels=labels, ylabel="Absolute peak error (pp)", title="Peak error by patient")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    tt, true, affected = example["times"], example["truth"], example["hidden"]
    curve.plot(tt, true, linestyle="--", lw=1.1, color="#303840", label="Complete measured curve")
    curve.scatter(tt[affected], true[affected], color="#303840", s=14, marker="x", label="Affected reference frames")
    for name, values in example["predictions"].items():
        color = ablation.COLORS[name]
        curve.plot(tt, values, color=color, linewidth=1, alpha=0.9, label=ablation.MODEL_LABELS[name])
        idx = int(np.argmax(values))
        curve.scatter(tt[idx], values[idx], facecolor="none", edgecolor=color, s=42, linewidth=1.2, zorder=4)
    curve.set(xlabel="TT (s)", ylabel="Change vs. reference (%)", title=f"{example['sample']} | {example['unit']}")
    curve.grid(alpha=0.2)
    curve.legend(frameon=False, fontsize=7)
    fig.savefig(output / "peak_comparison.png", dpi=145)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--roi-manifest", type=Path, help="Optional CSV for verified lesion masks")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive.")
    device, _ = training.configure_device(parser, args)
    try:
        reference, visits, fingerprint = ablation.load_reference()
        control = json.loads((ablation.OUTPUT / "run.json").read_text(encoding="utf-8"))
        if (control.get("version") != 1 or control.get("base_run_id") != reference["run_id"] or
                control.get("inputs") != fingerprint):
            raise ValueError("06 outputs do not match the current 04 run and data. Run 06 before 07.")
        controls = {fold["fold"]: fold for fold in control["folds"]}
        for fold in reference["folds"]:
            if any(fold[role] != controls[fold["fold"]][role] for role in ("train", "validation", "test")):
                raise ValueError("04 and 06 use different patient splits.")
        rois = roi_rows(args.roi_manifest.resolve(), visits) if args.roi_manifest else None
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))

    units, visit_rows, roi_qc = [], [], []
    examples = {}
    for fold in reference["folds"]:
        graph_model, time_model = ablation.load_fold_models(reference, fold, control, controls[fold["fold"]], device)
        with torch.inference_mode():
            for visit in visits:
                if visit.patient not in fold["test"] or (rois is not None and visit.sample not in rois):
                    continue
                case = visit.cases[MASK]
                print(f"Peak evaluation | fold {fold['fold']} | {visit.sample}", flush=True)
                device_visit = training.visit_to(visit, device)
                predictions = ablation.predictions(visit, device_visit, MASK, graph_model, time_model)
                if rois is None:
                    truth, hidden = case.truth, case.hidden.numpy()
                    labels = list(range(len(truth)))
                else:
                    lesion, coverage = lesion_curves(visit, case, predictions, rois[visit.sample])
                    status = "foreground coverage below 95%" if lesion is None else "eligible"
                    if lesion is not None and not lesion[2].any():
                        status = "no overlap with original local gap"
                    roi_qc.append(dict(sample=visit.sample, foreground_coverage=coverage, status=status))
                    if status != "eligible":
                        continue
                    truth, predictions, hidden, labels = lesion
                visit_units = peak_metrics(truth, predictions, hidden, labels)
                common = dict(patient=visit.patient, visit=visit.visit, sample=visit.sample, fold=fold["fold"])
                units.extend(dict(**common, **row) for row in visit_units)
                peak_gap = [row for row in visit_units if row["peak_affected"]]
                for subset, selected in (("all_affected", visit_units), ("peak_affected", peak_gap)):
                    if selected:
                        visit_rows.append(dict(**common, analysis=subset, units=len(selected), **{
                            metric: float(np.mean([row[f"{method}_peak_error_pp"] for row in selected]))
                            for method, metric in zip(METHODS, METRICS)
                        }))
                if visit_units:
                    row = (peak_gap or visit_units)[0]
                    i = labels.index(row["unit"])
                    examples[visit.sample] = dict(sample=visit.sample, unit=row["unit"], times=visit.times,
                                                  truth=truth[i], hidden=hidden[i], peak_affected=row["peak_affected"],
                                                  predictions={name: values[i] for name, values in predictions.items()})

    output = OUTPUT / "lesion" if rois is not None else OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    if rois is not None:
        audit.write_csv(output / "roi_qc.csv", roi_qc)
    if not units:
        parser.error(f"No eligible local-gap cases. In lesion mode, check {output / 'roi_qc.csv'}.")
    patients, summary = ablation.macro_average(visit_rows, METRICS)
    audit.write_csv(output / "peak_errors_by_unit.csv", units)
    audit.write_csv(output / "metrics_by_visit.csv", visit_rows)
    audit.write_csv(output / "metrics_by_patient.csv", patients)
    audit.write_csv(output / "summary.csv", summary)
    example = min(examples.values(), key=lambda e: (not e["peak_affected"], e["sample"]))
    save_plot(output, patients, example)
    table = audit.html_table(["分析范围", "患者数", "线性峰值误差", "只看本区域", "GNN", "GNN − 线性"], [
        (SUBSETS[r["analysis"]], r["patients"], f"{r['linear_peak_mae_pp']:.3f}", f"{r['time_only_peak_mae_pp']:.3f}",
         f"{r['gnn_peak_mae_pp']:.3f}", f"{r['gnn_peak_mae_pp'] - r['linear_peak_mae_pp']:+.3f}") for r in summary
    ])
    scope = ("使用外部确认与 01 切片对齐的病灶掩码。用各节点与掩码重合的像素数做权重，汇成一条区域平均曲线。"
             "这是病灶覆盖区域的超像素级信号代理。仅纳入掩码至少 95% 落在图前景内且与原局部缺口有交集的访视。"
             '<a href="roi_qc.csv">ROI 纳入和排除记录</a>。') if rois is not None else (
             "本轮做的是普通区域分析。图中的每个节点是一条区域曲线，目前没有使用病灶标注。")
    training.graphs.prepare.write_report(output, "07 · 补完以后，峰值也更准了吗？", f"""
<section><h2>测什么</h2><p>{scope}</p>
<p>每种方法使用相同的局部缺失点。分别取完整曲线和补全曲线在现有采样时刻的最大值，再算两者相差多少个百分点。比如完整峰值为 +70%，补完算出 +62%，差 8 个百分点。这里评估的是相对首帧信号的采样峰高。</p>
<p>第一行纳入所有在原掩码下受影响的曲线。第二行只保留参考峰值落在受影响帧里的曲线，方便看真正可能漏掉峰值的情况。若多个时刻并列峰值，只有全部受影响才进第二行。先平均同次访视的入选曲线，再平均同一患者的访视，最后对患者等权。</p>{table}</section>
<section><h2>怎么看图</h2><figure><img src="peak_comparison.png" alt="逐患者峰值误差和留出患者的峰值示例">
<figcaption>左图的小点代表患者，菱形是各方法平均，越低越好。右图黑虚线是完整实测曲线，叉号标出受影响的时刻，三种颜色分别对应补完后的曲线，空心圈标记各自的峰值。示例优先选参考峰值受影响的曲线，按样本和节点编号固定选择。</figcaption></figure>
<p><a href="peak_errors_by_unit.csv">逐区域或病灶峰值</a> · <a href="metrics_by_patient.csv">逐患者结果</a> · <a href="summary.csv">汇总表</a>。</p></section>
<section><p>这些结果沿用 05 已查看过的同一批留出患者，是针对本批数据和人工缺失方式的探索性分析。医学解释需要经核验的病灶标注和独立验证。</p></section>""")
    print(f"Report: {output / 'index.html'}")


if __name__ == "__main__":
    main()
