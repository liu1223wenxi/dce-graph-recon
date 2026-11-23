"""Extract one spatially matched DCE slice per visit and normalize its signal."""

import argparse
from html import escape
from importlib import import_module
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import label

matplotlib.use("Agg")
from matplotlib import pyplot as plt

audit = import_module("00_inspect_data")
OUTPUTS = audit.PROJECT_ROOT / "outputs"
PREPARED_DIR = OUTPUTS / "01_prepared"


def write_report(output: Path, title: str, body: str) -> None:
    (output / "index.html").write_text(f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title><style>
body {{ max-width:1000px; margin:2rem auto; padding:0 1rem; font:16px/1.65 system-ui,sans-serif; color:#17304a; background:#f5f8fb; }}
section {{ margin:1rem 0; padding:.75rem 1.25rem; border:1px solid #dae4ee; border-radius:12px; background:white; }}
img {{ max-width:100%; height:auto; display:block; }} table {{ border-collapse:collapse; }}
.table-wrap {{ overflow-x:auto; }} th, td {{ text-align:left; border-bottom:1px solid #e0e8ee; padding:.35rem .6rem; }}
figure {{ margin:1rem 0; }} figcaption {{ color:#354b5f; }} a {{ color:#145f78; }}
</style><body><h1>{escape(title)}</h1>{body}</body></html>""", encoding="utf-8")


def load_visit_slice(root: Path, ordered_frames: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    # Use physical slice positions from 00, rather than filename or instance order.
    first = audit.slice_index(root, ordered_frames[0])
    ds = audit.dcmread(first[0][1], stop_before_pixels=True, specific_tags=audit.HEADER_TAGS)
    directions = np.asarray(ds.ImageOrientationPatient, dtype=float).reshape(2, 3)
    normal = np.cross(directions[0], directions[1])
    first.sort(key=lambda item: item[0] @ normal)
    middle = (len(first) - 1) // 2
    position, first_path = first[middle]
    frames = [audit.load_slice(first_path)]
    for row in ordered_frames[1:]:
        matches = [
            path for point, path in audit.slice_index(root, row)
            if np.allclose(point, position, atol=audit.POSITION_TOL_MM, rtol=0)
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one physical slice at TT={row['tt_seconds']} s in {row['series_uid']}.")
        frames.append(audit.load_slice(matches[0]))
    return np.stack(frames), np.asarray(ds.PixelSpacing, dtype=float), position, middle + 1


def prepare_signal(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    reference = raw[0]
    if not np.isfinite(raw).all():
        raise ValueError("Decoded MR signal contains NaN or inf.")
    positive = reference[reference > 0]
    if positive.size == 0:
        raise ValueError("Reference frame has no positive signal.")
    threshold = 0.08 * float(np.percentile(positive, 98))
    # Foreground is estimated from the retained reference alone. Remove isolated speckle.
    components, _ = label(reference > threshold)
    counts = np.bincount(components.ravel())
    keep = counts >= 64
    keep[0] = False
    mask = keep[components]
    if np.count_nonzero(mask) < 64:
        raise ValueError("Too little foreground in the selected slice.")
    relative_pct = np.zeros_like(raw, dtype=np.float32)
    relative_pct[:, mask] = 100 * (raw[:, mask] - reference[mask]) / reference[mask]
    return relative_pct, mask, threshold


def save_preview(output: Path, sample: str, raw: np.ndarray, mask: np.ndarray,
                 relative_pct: np.ndarray, times: np.ndarray) -> None:
    index = len(times) // 3
    change = np.ma.masked_where(~mask, relative_pct[index])
    limit = max(1, float(np.percentile(np.abs(change.compressed()), 98)))
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), layout="constrained")
    axes[0].imshow(raw[0], cmap="gray", vmin=0, vmax=np.percentile(raw[0][mask], 99))
    axes[0].set_title(f"Reference | TT {times[0]:g} s")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Reference-derived foreground")
    image = axes[2].imshow(change, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[2].set_title(f"Relative change | TT {times[index]:g} s")
    fig.colorbar(image, ax=axes[2], label="Change vs. reference (%)", shrink=0.75)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(sample)
    fig.savefig(output / "preview.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--patient", help="Optional short patient ID, e.g. BC05")
    args = parser.parse_args()
    root = args.data_root.resolve()
    if PREPARED_DIR.resolve().is_relative_to(root):
        parser.error("The raw data path must not contain the output directory.")
    qc_dir = OUTPUTS / "00_data_audit"
    if not all((qc_dir / name).is_file() for name in ("studies.csv", "mr_series.csv")):
        parser.error("Run 00_inspect_data.py to create the audit CSVs first.")
    studies = audit.read_csv(qc_dir / "studies.csv")
    frames_by_study = {}
    for row in audit.read_csv(qc_dir / "mr_series.csv"):
        key = (row["patient_id"], row["study_uid"])
        frames_by_study.setdefault(key, []).append(row)

    selected = []
    excluded = []
    for row in studies:
        patient = row["patient_id"].rsplit("-", 1)[-1]
        if args.patient and patient.upper() != args.patient.upper():
            continue
        visit = audit.visit_label(row)
        key = (row["patient_id"], row["study_uid"])
        if (not visit or int(row["series_with_tt"]) < 4 or int(row["mr_series_flagged"])
                or int(row["duplicate_tt_labels"])):
            excluded.append((patient, visit, row["study_uid"], "复查访视标签、时间标签或几何 QC"))
            continue
        frames = sorted(frames_by_study[key], key=lambda frame: float(frame["tt_seconds"]))
        selected.append((patient, visit, row["study_uid"], frames))
    if not selected:
        parser.error("No visits passed 00 audit for the selected patient(s).")
    sample_ids = [f"{patient}_V{visit}" for patient, visit, _, _ in selected]
    if len(set(sample_ids)) != len(selected):
        parser.error("Duplicate patient/visit combination in the audit.")

    sample_dir = root / "derived" / "dce-graph-recon" / "01_prepared" / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, (patient, visit, study_uid, frames) in enumerate(sorted(selected)):
        sample = f"{patient}_V{visit}"
        print(f"[{index + 1}/{len(selected)}] Preparing {sample}", flush=True)
        raw, spacing, position, slice_number = load_visit_slice(root, frames)
        times = np.asarray([float(row["tt_seconds"]) for row in frames])
        relative_pct, mask, threshold = prepare_signal(raw)
        sample_path = sample_dir / f"{sample}.npz"
        np.savez_compressed(sample_path, times_s=times, signal=raw,
                            relative_pct=relative_pct, foreground=mask, spacing_mm=spacing,
                            slice_position_mm=position)
        manifest.append(dict(sample=sample, patient=patient, visit=visit, study_uid=study_uid,
                             slice_number=slice_number, frames=len(times), reference_tt_s=times[0],
                             rows=mask.shape[0], columns=mask.shape[1], foreground_pixels=int(mask.sum()),
                             foreground_threshold=threshold, file=sample_path.as_posix()))
        if index == 0:
            save_preview(PREPARED_DIR, sample, raw, mask, relative_pct, times)

    audit.write_csv(PREPARED_DIR / "manifest.csv", manifest)
    table = audit.html_table(["样本", "切片编号", "时间帧", "首帧 TT (s)", "前景像素"],
                             [(r["sample"], r["slice_number"], r["frames"], r["reference_tt_s"],
                               r["foreground_pixels"]) for r in manifest])
    exclusions = audit.html_table(["患者", "访视", "检查 UID", "原因"], excluded) if excluded else "本次筛查没有 QC 排除。"
    write_report(PREPARED_DIR, "01 · 从 DICOM 到时空数组", f"""
<section><p>每次检查提取中央轴位切片，按物理坐标在每个时间帧找到同一层。一个样本就是 <strong>[时间, 行, 列]</strong> 的数组。</p>
<figure><img src="preview.png" alt="MRI 首帧、前景与相对信号变化"><figcaption>左图用于确定前景。中图的白色区域会进入建图。右图显示相对首帧的信号变化，极端值仅在显示时截断。</figcaption></figure>
<p>相对信号变化 = 100 × (当前信号 − 首帧信号) / 首帧信号。首帧会保留为输入，且这里尚未验证它是否为注射前影像。
前景是从首帧信号估计的区域。它不提供病灶定位，本阶段也没有做运动配准。</p></section>
<section>{table}<p><a href="manifest.csv">样本清单和 NPZ 路径</a></p><h2>排除记录</h2>{exclusions}</section>""")
    print(f"Report: {PREPARED_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
