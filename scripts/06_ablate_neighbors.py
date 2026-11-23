"""Compare the trained GNN with a matched time-only model and unavailable neighbor measurements."""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from importlib import import_module

import numpy as np
import torch
from torch import nn

evaluate = import_module("05_evaluate")
training, audit, plt = evaluate.training, evaluate.audit, evaluate.plt
OUTPUT = training.graphs.prepare.OUTPUTS / "06_ablation"
MASK_ZH = evaluate.MASK_ZH
MODEL_LABELS = {"linear": "Linear", "time_only": "Time-only", "gnn": "GNN", "no_same_time": "No same-time neighbor"}
COLORS = {"linear": "#2088a6", "time_only": "#c55b58", "gnn": "#8259a4", "no_same_time": "#686e77"}


class TimeOnlyResidual(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(training.FEATURE_COUNT, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, visit, case):
        linear = case.features[..., 0]
        return torch.where(case.hidden, linear + self.mlp(case.features).squeeze(-1), linear)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_reference():
    record = json.loads((training.TRAINING / "run.json").read_text(encoding="utf-8"))
    visits, fingerprint = training.load_data()
    evaluate.check_splits(record, {visit.patient for visit in visits})
    if record.get("version") != 1 or record.get("inputs") != fingerprint:
        raise ValueError("04 run and current graphs/masks do not match. Use the same run and data as 05.")
    return record, visits, fingerprint


def load_fold_models(reference: dict, original: dict, ablation: dict, control: dict, device):
    graph_model = training.NeighborGNN(reference["parameters"]["width"])
    time_model = TimeOnlyResidual(ablation["time_width"])
    graph_checkpoint = training.TRAINING / "runs" / reference["run_id"] / original["checkpoint"]
    time_checkpoint = OUTPUT / "runs" / ablation["run_id"] / control["checkpoint"]
    for model, path in ((graph_model, graph_checkpoint), (time_model, time_checkpoint)):
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.to(device).eval()
    return graph_model, time_model


def predictions(visit, device_visit, mask: str, graph_model, time_model) -> dict[str, np.ndarray]:
    case, device_case = visit.cases[mask], device_visit.cases[mask]
    hidden = case.hidden.numpy()
    outputs = {"linear": case.linear}
    for name, model in (("time_only", time_model), ("gnn", graph_model)):
        value = model(device_visit, device_case).cpu().numpy() * training.SIGNAL_SCALE
        outputs[name] = np.where(hidden, value, case.linear)
    return outputs


def neighbor_intervention(visit, case) -> tuple[torch.Tensor, np.ndarray]:
    """Replace only source-node features at t, using observed brackets strictly before/after t."""
    hidden = case.hidden.numpy()
    observed = ~hidden
    src, dst = visit.src.numpy(), visit.dst.numpy()
    # An affected target must itself be hidden and have an incoming observed source at that time.
    neighbor_count = np.zeros_like(hidden, dtype=int)
    np.add.at(neighbor_count, dst, observed[src])
    exposed = hidden & (neighbor_count > 0)
    source = case.features.numpy().copy()
    nodes, ticks = np.nonzero(observed & hidden.any(axis=0)[None, :])
    if not len(ticks):
        return torch.from_numpy(source), exposed
    indices = np.broadcast_to(np.arange(len(visit.times)), hidden.shape)
    left = np.maximum.accumulate(np.where(observed, indices, -1), axis=1)[nodes, ticks - 1]
    right = np.minimum.accumulate(np.where(observed, indices, len(visit.times))[:, ::-1], axis=1)[:, ::-1]
    right = right[nodes, ticks + 1]
    times = visit.times
    t_left, t_right, t = times[left], times[right], times[ticks]
    y_left = case.linear[nodes, left]
    y_right = case.linear[nodes, right]
    delta = y_right - y_left
    source[nodes, ticks, 0] = (y_left + (t - t_left) / (t_right - t_left) * delta) / training.SIGNAL_SCALE
    source[nodes, ticks, 1] = 0
    source[nodes, ticks, 3] = (t - t_left) / 60
    source[nodes, ticks, 4] = (t_right - t) / 60
    source[nodes, ticks, 5] = delta / training.SIGNAL_SCALE
    return torch.from_numpy(source), exposed


def macro_average(visits: list[dict], metrics: tuple[str, ...]) -> tuple[list[dict], list[dict]]:
    by_patient = defaultdict(list)
    for row in visits:
        by_patient[row["analysis"], row["patient"]].append(row)
    patients = [dict(analysis=analysis, patient=patient, visits=len(rows),
                     **{key: float(np.mean([r[key] for r in rows])) for key in metrics})
                for (analysis, patient), rows in sorted(by_patient.items())]
    summaries = []
    for analysis in dict.fromkeys(row["analysis"] for row in visits):
        subset = [r for r in patients if r["analysis"] == analysis]
        summaries.append(dict(analysis=analysis, patients=len(subset),
                              **{key: float(np.mean([r[key] for r in subset])) for key in metrics}))
    return patients, summaries


def save_plot(patients: list[dict], diagnostic: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), layout="constrained")
    panels = [(mask, [r for r in patients if r["analysis"] == mask], ["linear", "time_only", "gnn"])
              for mask in MASK_ZH]
    panels.append(("neighbor_intervention", diagnostic, ["gnn", "no_same_time"]))
    for ax, (name, rows, models) in zip(axes, panels):
        for row in rows:
            ax.plot(range(len(models)), [row[f"{method}_mae_pp"] for method in models],
                    marker="o", markersize=3, color="#63768c", alpha=0.52)
        if rows:
            means = [np.mean([r[f"{method}_mae_pp"] for r in rows]) for method in models]
            ax.scatter(range(len(models)), means, c=[COLORS[m] for m in models], s=55, zorder=4)
        title = evaluate.MASK_LABELS.get(name, "Local gap: observed-neighbor subset")
        ticks = [MODEL_LABELS[m] if m != "no_same_time" else "GNN with\nmeasurements removed" for m in models]
        ax.set(xticks=range(len(models)), xticklabels=ticks, title=title, ylabel="Hidden-point MAE (pp)")
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(OUTPUT / "ablation_comparison.png", dpi=145)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    cli = parser.parse_args()
    if cli.threads < 1:
        parser.error("--threads must be positive.")
    device, device_name = training.configure_device(parser, cli)
    try:
        reference, visits, fingerprint = load_reference()
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))

    params = reference["parameters"]
    graph_count = parameter_count(training.NeighborGNN(params["width"]))
    # For this control architecture the count is w² + (FEATURE_COUNT + 3)w + 1.
    width = min(range(1, 2 * params["width"] + 2),
                key=lambda w: abs(w * w + (training.FEATURE_COUNT + 3) * w + 1 - graph_count))
    time_count = parameter_count(TimeOnlyResidual(width))
    args = argparse.Namespace(**(params | {"width": width, "device": cli.device, "threads": cli.threads}))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = OUTPUT / "runs" / run_id
    run_dir.mkdir(parents=True)
    grouped = training.patient_cases([training.visit_to(v, device) for v in visits])
    record = dict(version=1, run_id=run_id, base_run_id=reference["run_id"], inputs=fingerprint,
                  device=device_name, time_width=width, graph_parameters=graph_count,
                  time_parameters=time_count, folds=[])
    visit_rows, diagnostic_rows, history = [], [], []
    for original in reference["folds"]:
        split = {name: original[name] for name in ("fold", "train", "validation", "test")}
        print(f"Fold {split['fold']} | training time-only model", flush=True)
        epochs, control = training.train_fold(split, grouped, run_dir, args, device, model_class=TimeOnlyResidual)
        record["folds"].append(control)
        history.extend(epochs)
        graph_model, time_model = load_fold_models(reference, original, record, control, device)
        with torch.inference_mode():
            for visit in visits:
                if visit.patient not in split["test"]:
                    continue
                device_visit = training.visit_to(visit, device)
                common = dict(sample=visit.sample, patient=visit.patient, visit=visit.visit, fold=split["fold"])
                for mask, case in visit.cases.items():
                    hidden = case.hidden.numpy()
                    estimates = predictions(visit, device_visit, mask, graph_model, time_model)
                    mae = {f"{method}_mae_pp": evaluate.scores(value, case.truth, hidden)[0]
                           for method, value in estimates.items()}
                    visit_rows.append(dict(**common, analysis=mask, hidden_points=int(hidden.sum()), **mae))
                    if mask != "local_gap":
                        continue
                    source_features, exposed = neighbor_intervention(visit, case)
                    if exposed.any():
                        value = graph_model(device_visit, device_visit.cases[mask],
                                            source_features=source_features.to(device)).cpu().numpy() * training.SIGNAL_SCALE
                        diagnostic_rows.append(dict(**common, analysis="neighbor_intervention",
                                                    evaluated_points=int(exposed.sum()),
                                                    gnn_mae_pp=evaluate.scores(estimates["gnn"], case.truth, exposed)[0],
                                                    no_same_time_mae_pp=evaluate.scores(value, case.truth, exposed)[0]))

    patients, summary = macro_average(visit_rows, ("linear_mae_pp", "time_only_mae_pp", "gnn_mae_pp"))
    diag_patients, diag_summary = macro_average(diagnostic_rows, ("gnn_mae_pp", "no_same_time_mae_pp"))
    audit.write_csv(OUTPUT / "metrics_by_visit.csv", visit_rows)
    audit.write_csv(OUTPUT / "metrics_by_patient.csv", patients)
    audit.write_csv(OUTPUT / "summary.csv", summary)
    audit.write_csv(run_dir / "history.csv", history)
    if diagnostic_rows:
        audit.write_csv(OUTPUT / "neighbor_intervention_by_visit.csv", diagnostic_rows)
        audit.write_csv(OUTPUT / "neighbor_intervention_by_patient.csv", diag_patients)
    save_plot(patients, diag_patients)
    table = audit.html_table(["缺失方式", "患者数", "线性", "只看本区域", "GNN", "GNN − 只看本区域"], [
        (MASK_ZH[r["analysis"]], r["patients"], f"{r['linear_mae_pp']:.3f}", f"{r['time_only_mae_pp']:.3f}",
         f"{r['gnn_mae_pp']:.3f}", f"{r['gnn_mae_pp'] - r['time_only_mae_pp']:+.3f}") for r in summary
    ])
    diagnostic = audit.html_table(["患者数", "原 GNN", "去掉邻居同刻测量", "误差变化"], [
        (r["patients"], f"{r['gnn_mae_pp']:.3f}", f"{r['no_same_time_mae_pp']:.3f}",
         f"{r['no_same_time_mae_pp'] - r['gnn_mae_pp']:+.3f}") for r in diag_summary
    ]) if diag_summary else "<p>当前掩码下没有符合条件的点。</p>"
    training.graphs.prepare.write_report(OUTPUT, "06 · 邻居到底贡献了什么？", f"""
<section><h2>对照一：让一个同样会学习的模型只看本区域</h2>
<p>线性插值给出初值。只看本区域的模型用与 GNN 相同的目标节点特征学习修正，但没有邻居和边特征。两种模型分别有 {time_count} 和 {graph_count} 个参数，沿用 04 的患者分折、训练预算和选轮规则。GNN 直接用 04 保存的最佳权重。误差只在原始隐藏点上计算，先平均同一患者的访视，再等权平均患者。</p>
{table}<p>单位均为百分点。最后一列为负，表示 GNN 比只看本区域的模型更准。</p></section>
<section><h2>对照二：保持缺口不变，只撤掉邻居当时的实测值</h2>
<p>只看局部缺失中原本有实测邻居的点。边、目标的可见数据和评估位置保持一致。同一个 GNN 再预测一次，这次每条来自实测邻居的消息改用该邻居前后观测算出的线性估计，同时更新观测标记和时间缺口特征。这是推理时的敏感性检查，使用了比训练时更少的同刻测量。</p>
{diagnostic}<p>误差变化为正，表示撤掉这类观测后估计更差。此表只统计受干预的点。</p></section>
<section><figure><img src="ablation_comparison.png" alt="逐患者配对误差"><figcaption>灰线连的是同一位患者。彩点是患者平均。前两格比较三种方法，右格只比较受到干预的局部缺失点。横轴方法变了，纵轴仍然是 MAE。</figcaption></figure>
<p><a href="metrics_by_visit.csv">访视明细</a> · <a href="metrics_by_patient.csv">逐患者对照</a> · <a href="summary.csv">汇总</a> · <a href="runs/{run_id}/history.csv">对照模型训练记录</a></p></section>""")
    (OUTPUT / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Report: {OUTPUT / 'index.html'}")


if __name__ == "__main__":
    main()
