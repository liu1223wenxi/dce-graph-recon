"""Fit a one-hop residual GNN with patient-grouped cross-validation."""

import argparse
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import file_digest
from html import escape
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from torch import nn

baseline = import_module("03_run_baselines")
graphs, audit, plt = baseline.graphs, baseline.audit, baseline.plt
TRAINING = graphs.prepare.OUTPUTS / "04_training"
FEATURE_COUNT = 7
SIGNAL_SCALE = 100.0  # Model uses fractions. Reports use percentage points.


@dataclass
class Case:
    truth: np.ndarray
    linear: np.ndarray
    hidden: torch.Tensor
    features: torch.Tensor
    target: torch.Tensor


@dataclass
class Visit:
    sample: str
    patient: str
    visit: str
    graph_path: Path
    times: np.ndarray
    src: torch.Tensor
    dst: torch.Tensor
    distance: torch.Tensor
    cases: dict[str, Case]


def configure_device(parser: argparse.ArgumentParser, args) -> tuple[torch.device, str]:
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this environment. Install a CUDA build of PyTorch or use --device cpu.")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    torch.set_num_threads(args.threads)
    label = f"{device} ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "cpu"
    print(f"PyTorch device: {label}", flush=True)
    return device, label


def visit_to(visit: Visit, device: torch.device) -> Visit:
    # Keep NumPy arrays and metadata on the host. Only model tensors move.
    if device.type == "cpu":
        return visit
    cases = {
        name: replace(case, hidden=case.hidden.to(device), features=case.features.to(device),
                      target=case.target.to(device))
        for name, case in visit.cases.items()
    }
    return replace(visit, src=visit.src.to(device), dst=visit.dst.to(device),
                   distance=visit.distance.to(device), cases=cases)


def prepare_case(times: np.ndarray, truth: np.ndarray, hidden: np.ndarray,
                 reference: np.ndarray) -> Case:
    if hidden.dtype != bool or hidden.shape != truth.shape or not hidden.any():
        raise ValueError("Expected a nonempty Boolean mask matching the graph's signal matrix.")
    if hidden[:, [0, -1]].any():
        raise ValueError("The 03 protocol must retain the first and last observations of every node.")
    # Build model inputs only from unmasked signals and the retained reference image.
    observed = np.where(hidden, np.nan, truth).astype(float)
    linear, _ = baseline.time_interpolation(times, observed, hidden)
    indices = np.broadcast_to(np.arange(len(times)), hidden.shape)
    left = np.maximum.accumulate(np.where(~hidden, indices, -1), axis=1)
    right = np.minimum.accumulate(np.where(~hidden, indices, len(times))[:, ::-1], axis=1)[:, ::-1]
    time_fraction = (times - times[0]) / (times[-1] - times[0])
    relative_ref = np.log(np.maximum(reference, 1e-6) / max(float(np.median(reference)), 1e-6))
    features = np.stack([
        linear / SIGNAL_SCALE,
        (~hidden).astype(float),
        np.broadcast_to(time_fraction, hidden.shape),
        (times - times[left]) / 60.0,
        (times[right] - times) / 60.0,
        (np.take_along_axis(observed, right, axis=1) - np.take_along_axis(observed, left, axis=1)) / SIGNAL_SCALE,
        np.broadcast_to(relative_ref[:, None], hidden.shape),
    ], axis=-1)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite model features. Check observations and timestamps.")
    return Case(truth, linear, torch.from_numpy(hidden), torch.tensor(features, dtype=torch.float32),
                torch.tensor(truth / SIGNAL_SCALE, dtype=torch.float32))


def load_data() -> tuple[list[Visit], dict]:
    manifest = graphs.GRAPH_DIR / "manifest.csv"
    if not manifest.is_file():
        raise ValueError("Run 02_build_graphs.py before training.")
    rows = sorted(audit.read_csv(manifest), key=lambda row: row["sample"])
    if not rows or len({row["sample"] for row in rows}) != len(rows):
        raise ValueError("The 02 manifest is empty or contains duplicate samples.")
    visits, fingerprint = [], {}
    for row in rows:
        sample = row["sample"]
        graph_path = graphs.GRAPH_DIR / row["file"]
        mask_path = baseline.RESULTS / "masks" / f"{sample}.npz"
        if not mask_path.is_file():
            raise ValueError(f"Missing 03 masks for {sample}. Run 03_run_baselines.py first.")
        with graph_path.open("rb") as gf, mask_path.open("rb") as mf:
            fingerprint[sample] = dict(patient=row["patient"], visit=row["visit"],
                                       graph_sha256=file_digest(gf, "sha256").hexdigest(),
                                       mask_sha256=file_digest(mf, "sha256").hexdigest())
        with np.load(graph_path, allow_pickle=False) as graph, np.load(mask_path, allow_pickle=False) as masks:
            times, truth = graph["times_s"], graph["signals_pct"]
            if len(times) < 4 or not np.all(np.diff(times) > 0) or not np.isfinite(truth).all():
                raise ValueError(f"Invalid times or signals in {sample}.")
            if not np.array_equal(times, masks["times_s"]):
                raise ValueError(f"02 graph and 03 masks have different times for {sample}.")
            edge = graph["edges"]
            if edge.ndim != 2 or edge.shape[1] != 2 or np.any(edge < 0) or np.any(edge >= len(truth)):
                raise ValueError(f"Invalid edge array in {sample}.")
            src = torch.tensor(np.r_[edge[:, 0], edge[:, 1]], dtype=torch.long)
            dst = torch.tensor(np.r_[edge[:, 1], edge[:, 0]], dtype=torch.long)
            distance = torch.tensor(np.tile(graph["edge_distance_mm"], 2) / 20.0, dtype=torch.float32)
            reference = graph["node_reference_mean"]
            cases = {name: prepare_case(times, truth, masks[name], reference) for name in baseline.MASK_LABELS}
        visits.append(Visit(sample, row["patient"], row["visit"], graph_path, times, src, dst, distance, cases))
    return visits, fingerprint


class NeighborGNN(nn.Module):
    """Each directed edge produces a gate and signed correction at each time."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(2 * FEATURE_COUNT + 1, width), nn.SiLU(),
                                      nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2))
        # Start with exactly the linear interpolation estimate.
        nn.init.zeros_(self.edge_mlp[-1].weight)
        nn.init.zeros_(self.edge_mlp[-1].bias)

    def forward(self, visit: Visit, case: Case, explain: bool = False,
                source_features: torch.Tensor | None = None):
        x, src, dst = case.features, visit.src, visit.dst
        linear = x[..., 0]
        edge_distance = visit.distance[:, None, None].expand(-1, x.shape[1], 1)
        neighbor_x = x if source_features is None else source_features
        pairs = torch.cat((x[dst], neighbor_x[src], edge_distance), dim=-1)
        edge_output = self.edge_mlp(pairs)
        gates = torch.sigmoid(edge_output[..., 0]) + 1e-6
        gate_sum = torch.zeros_like(linear).index_add(0, dst, gates)
        weights = gates / gate_sum[dst].clamp_min(1e-6)
        contribution = weights * edge_output[..., 1]
        correction = torch.zeros_like(linear).index_add(0, dst, contribution)
        prediction = torch.where(case.hidden, linear + correction, linear)
        return (prediction, weights, contribution) if explain else prediction


def patient_folds(patients: list[str], folds: int, seed: int) -> list[dict]:
    if not 3 <= folds <= len(patients):
        raise ValueError(f"Need 3 to {len(patients)} folds to have separate train, validation, and test patients.")
    shuffled = np.random.default_rng(seed).permutation(sorted(patients))
    groups = [set(group.tolist()) for group in np.array_split(shuffled, folds)]
    return [dict(fold=k + 1, test=sorted(test), validation=sorted(groups[(k + 1) % folds]),
                 train=sorted(set(patients) - test - groups[(k + 1) % folds]))
            for k, test in enumerate(groups)]


def patient_cases(visits: list[Visit]) -> dict[str, list[tuple[Visit, Case]]]:
    grouped = {}
    for visit in visits:
        grouped.setdefault(visit.patient, []).extend((visit, case) for case in visit.cases.values())
    return grouped


def hidden_mae(model: nn.Module, visit: Visit, case: Case):
    return (model(visit, case)[case.hidden] - case.target[case.hidden]).abs().mean()


def train_fold(split: dict, grouped: dict, run_dir: Path, args, device: torch.device,
               model_class=NeighborGNN) -> tuple[list[dict], dict]:
    fold = split["fold"]
    torch.manual_seed(args.seed + fold)
    rng = np.random.default_rng(args.seed + fold)
    model = model_class(args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_loss, best_epoch = float("inf"), 0
    history = []
    checkpoint = f"fold_{fold:02d}.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_errors = []
        for patient in rng.permutation(split["train"]):
            samples = grouped[patient]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for visit, case in samples:
                loss = hidden_mae(model, visit, case)
                (loss / len(samples)).backward()
                losses.append(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_errors.append(torch.stack(losses).mean())
        model.eval()
        with torch.no_grad():
            val_errors = [torch.stack([hidden_mae(model, v, c) for v, c in grouped[p]]).mean()
                          for p in split["validation"]]
        val_loss = torch.stack(val_errors).mean().item()
        train_loss = torch.stack(train_errors).mean().item()
        if not np.isfinite(val_loss) or not np.isfinite(train_loss):
            raise RuntimeError(f"Non-finite loss in fold {fold}, epoch {epoch}.")
        history.append(dict(fold=fold, epoch=epoch, train_mae_pp=train_loss * SIGNAL_SCALE,
                            validation_mae_pp=val_loss * SIGNAL_SCALE))
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            weights = {name: value.cpu() for name, value in model.state_dict().items()}
            torch.save(weights, run_dir / checkpoint)
        if epoch == 1 or epoch % 10 == 0:
            print(f"  Epoch {epoch:>3} | validation MAE {val_loss * SIGNAL_SCALE:.3f} pp", flush=True)
        if epoch - best_epoch >= args.patience:
            break
    print(f"  Selected epoch {best_epoch}, validation MAE {best_loss * SIGNAL_SCALE:.3f} pp", flush=True)
    return history, dict(**split, checkpoint=checkpoint, best_epoch=best_epoch,
                         validation_mae_pp=best_loss * SIGNAL_SCALE)


def save_training_plot(history: list[dict], folds: list[dict]) -> None:
    n = len(folds)
    fig, axes = plt.subplots((n + 2) // 3, min(3, n), figsize=(11, 3.2 * ((n + 2) // 3)),
                             squeeze=False, sharey=True, layout="constrained")
    for ax, split in zip(axes.flat, folds):
        rows = [r for r in history if r["fold"] == split["fold"]]
        epochs = [r["epoch"] for r in rows]
        ax.plot(epochs, [r["train_mae_pp"] for r in rows], color="#2088a6", label="Training")
        ax.plot(epochs, [r["validation_mae_pp"] for r in rows], color="#c55b58", label="Validation")
        ax.axvline(split["best_epoch"], color="#405266", linestyle="--", label="Saved epoch")
        ax.set(title=f"Fold {split['fold']}", xlabel="Epoch", ylabel="Hidden-point MAE (pp)")
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[n:]:
        ax.set_axis_off()
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.savefig(TRAINING / "training_curves.png", dpi=145)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40, help="Maximum per fold")
    parser.add_argument("--patience", type=int, default=8, help="Stop after this many epochs without improvement")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2, help="PyTorch CPU threads")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.width, args.threads) < 1 or args.learning_rate <= 0 or args.seed < 0:
        parser.error("Positive epochs, patience, width, learning rate and threads and a nonnegative seed are required.")
    device, device_label = configure_device(parser, args)
    try:
        visits, fingerprint = load_data()
        grouped = patient_cases([visit_to(visit, device) for visit in visits])
        splits = patient_folds(list(grouped), args.folds, args.seed)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = TRAINING / "runs" / run_id
    run_dir.mkdir(parents=True)
    history, folds = [], []
    for split in splits:
        print(f"Fold {split['fold']} | train {split['train']} | val {split['validation']} | test {split['test']}", flush=True)
        epochs, result = train_fold(split, grouped, run_dir, args, device)
        history.extend(epochs)
        folds.append(result)
    audit.write_csv(run_dir / "history.csv", history)
    audit.write_csv(run_dir / "patient_splits.csv", [
        dict(fold=f["fold"], role=role, patient=p) for f in folds
        for role in ("train", "validation", "test") for p in f[role]
    ])
    save_training_plot(history, folds)
    table = audit.html_table(["折", "训练患者", "验证患者", "测试患者", "保存轮次", "验证 MAE (百分点)"], [
        (f["fold"], ", ".join(f["train"]), ", ".join(f["validation"]), ", ".join(f["test"]),
         f["best_epoch"], f"{f['validation_mae_pp']:.2f}") for f in folds
    ])
    graphs.prepare.write_report(TRAINING, "04 · 学习怎样用邻居修正插值", f"""
<section><h2>想补哪一个值</h2>
<p>本轮训练设备为 <code>{escape(device_label)}</code>。</p>
<p>03 先把一部分测量藏起来。04 使用同一份掩码，先对每条区域曲线做线性插值。接下来，在每个缺失时刻问该节点的一圈直接邻居，应该把这个初值调高还是调低。</p>
<p>邻接边的两个方向各有一条消息。模型同时看目标和邻居的线性估计、该时刻是否实测、在扫描中的时间位置、离前后观测点的时间差、前后观测信号的差值、首帧相对亮度和区域中心距离。它为每条消息输出<strong>一个权重和一项可正可负的建议修正</strong>。同一目标收到的权重之和为 1。每条消息的贡献等于权重乘建议修正。比如初值是 +32%，两个邻居加权后的贡献是 +2 和 −1 个百分点，最终估计为 +33%。没有邻居的节点保留线性估计。</p>
<p>整帧缺失时，邻居在同一时刻只能提供从各自曲线算出的线性估计。局部缺失时，区域外邻居还能提供同一时刻的实测值。对已观测点直接保留测量。训练时只用被藏的测量值计算误差，逐折学习边上的权重与修正。</p></section>
<section><h2>患者怎么划分</h2>
<p>同一患者的两次访视总在同一组。每折用训练患者更新参数，再用独立的验证患者选保存轮次。测试患者的结果由 05 计算，每位患者恰好在一折担任测试。</p>
{table}<p><a href="runs/{run_id}/patient_splits.csv">完整的患者划分</a></p></section>
<section><h2>训练曲线怎么看</h2>
<figure><img src="training_curves.png" alt="各折训练和验证误差随轮次变化">
<figcaption>每格是一折。蓝线表示训练过程中各患者批次的误差，红线是在每轮结束后对验证患者测得的误差，纵轴只统计被隐藏的点。虚线处是红线最低的轮次，也是 05 要读取的权重。测试误差在 05 的配对图里。</figcaption></figure>
<p><a href="runs/{run_id}/history.csv">逐轮数值</a></p></section>""")
    record = dict(version=1, run_id=run_id, parameters=vars(args), device=device_label, inputs=fingerprint, folds=folds)
    (TRAINING / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Report: {TRAINING / 'index.html'}")


if __name__ == "__main__":
    main()
