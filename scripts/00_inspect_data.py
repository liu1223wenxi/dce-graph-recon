"""Audit and explore the local QIN Breast DCE-MRI download."""

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
from pydicom import dcmread
from pydicom.errors import InvalidDicomError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION = "qin_breast_dce_mri"
TIME_LABEL = re.compile(r"(?<![A-Za-z0-9])TT\s*=\s*(\d+(?:\.\d+)?)\s*s\b", re.IGNORECASE)
HEADER_TAGS = [
    "SeriesInstanceUID", "SOPInstanceUID", "Rows", "Columns", "PixelSpacing",
    "ImageOrientationPatient", "ImagePositionPatient",
]
POSITION_TOL_MM = 0.01
DIRECTION_TOL = 1e-4


@dataclass
class FrameGeometry:
    shape: tuple[int, int]
    spacing: np.ndarray
    orientation: np.ndarray
    positions: np.ndarray


def series_folder(data_root: Path, manifest_path: str) -> Path:
    """Rebase the manifest's download path if the collection has been moved."""
    parts = re.split(r"[\\/]+", manifest_path)
    try:
        index = next(i for i, part in enumerate(parts) if part.lower() == COLLECTION)
    except StopIteration as exc:
        raise ValueError(f"Collection name missing from series path: {manifest_path}") from exc
    collection_root = (data_root / COLLECTION).resolve()
    folder = collection_root.joinpath(*parts[index + 1:]).resolve()
    if folder == collection_root or not folder.is_relative_to(collection_root):
        raise ValueError(f"Invalid series path in manifest: {manifest_path}")
    return folder


def inspect_frame(files: list[Path], series_uid: str) -> tuple[FrameGeometry | None, str]:
    """Return header geometry and a QC message for a single MR series."""
    if not files:
        return None, "no DICOM files"

    positions = []
    instance_uids = set()
    try:
        for index, file in enumerate(files):
            ds = dcmread(file, stop_before_pixels=True, specific_tags=HEADER_TAGS)
            if str(ds.SeriesInstanceUID) != series_uid:
                return None, f"series UID mismatch in {file.name}"
            instance_uid = str(ds.SOPInstanceUID)
            if instance_uid in instance_uids:
                return None, f"duplicate SOP instance UID in {file.name}"
            instance_uids.add(instance_uid)

            current_shape = (int(ds.Rows), int(ds.Columns))
            current_spacing = np.asarray(ds.PixelSpacing, dtype=float)
            current_orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
            position = np.asarray(ds.ImagePositionPatient, dtype=float)
            if (
                min(current_shape) <= 0
                or current_spacing.shape != (2,)
                or not np.all(np.isfinite(current_spacing) & (current_spacing > 0))
                or current_orientation.shape != (6,)
                or not np.all(np.isfinite(current_orientation))
                or position.shape != (3,)
                or not np.all(np.isfinite(position))
            ):
                return None, f"invalid image geometry in {file.name}"

            if index == 0:
                shape = current_shape
                spacing = current_spacing
                orientation = current_orientation
            elif (
                current_shape != shape
                or not np.allclose(current_spacing, spacing, atol=POSITION_TOL_MM, rtol=0)
                or not np.allclose(current_orientation, orientation, atol=DIRECTION_TOL, rtol=0)
            ):
                return None, f"inconsistent slice geometry in {file.name}"
            positions.append(position)
    except (InvalidDicomError, OSError, AttributeError, ValueError, TypeError) as exc:
        return None, f"header read failed in {file.name}: {exc}"

    # Sort along the slice normal, independent of filename and scan orientation.
    row_direction, column_direction = orientation.reshape(2, 3)
    normal = np.cross(row_direction, column_direction)
    if not np.allclose(
        [np.linalg.norm(row_direction), np.linalg.norm(column_direction), row_direction @ column_direction],
        [1.0, 1.0, 0.0],
        atol=DIRECTION_TOL,
        rtol=0,
    ):
        return None, "invalid slice orientation"
    positions = np.stack(positions)
    slice_locations = positions @ normal
    order = np.argsort(slice_locations)
    gaps = np.diff(slice_locations[order])
    if np.any(gaps <= POSITION_TOL_MM):
        return None, "duplicate slice positions"
    qc = ""
    if not gaps.size:
        qc = "only one slice"
    elif not np.allclose(gaps, np.median(gaps), atol=POSITION_TOL_MM, rtol=0):
        qc = "uneven slice spacing"
    return FrameGeometry(shape, spacing, orientation, positions[order]), qc


def same_grid(first: FrameGeometry, other: FrameGeometry) -> bool:
    return (
        first.shape == other.shape
        and first.positions.shape == other.positions.shape
        and np.allclose(first.spacing, other.spacing, atol=POSITION_TOL_MM, rtol=0)
        and np.allclose(first.orientation, other.orientation, atol=DIRECTION_TOL, rtol=0)
        and np.allclose(first.positions, other.positions, atol=POSITION_TOL_MM, rtol=0)
    )


def audit_study(data_root: Path, rows: list[dict]) -> tuple[dict, list[dict]]:
    patient = rows[0]["PatientID"]
    study_uid = rows[0]["StudyInstanceUID"]
    auxiliary = [row for row in rows if row["Modality"] == "KO"]
    auxiliary_labels = []
    for row in auxiliary:
        folder = series_folder(data_root, row["S5cmdManifestPath"])
        if any(folder.glob("*.dcm")):
            auxiliary_labels.append(row["SeriesDescription"])

    frames = []
    for row in rows:
        if row["Modality"] != "MR":
            continue
        description = row["SeriesDescription"]
        match = TIME_LABEL.search(description)
        time_s = float(match.group(1)) if match else None
        frames.append((time_s, row))
    frames.sort(key=lambda item: float("inf") if item[0] is None else item[0])

    frame_rows = []
    reference = None
    for time_s, row in frames:
        folder = series_folder(data_root, row["S5cmdManifestPath"])
        files = sorted(folder.glob("*.dcm"))
        geometry, header_qc = inspect_frame(files, row["SeriesInstanceUID"])
        flags = [header_qc] if header_qc else []
        if time_s is None:
            flags.append("TT label unavailable")
        if geometry is not None:
            if reference is None:
                reference = geometry
            elif not same_grid(reference, geometry):
                flags.append("grid differs from first readable MR series")
        frame_rows.append({
            "patient_id": patient,
            "study_uid": study_uid,
            "series_uid": row["SeriesInstanceUID"],
            "series_number": row["SeriesNumber"],
            "series_description": row["SeriesDescription"],
            "series_folder": str(folder.relative_to(data_root)),
            "tt_seconds": time_s,
            "dicom_files": len(files),
            "rows": geometry.shape[0] if geometry else None,
            "columns": geometry.shape[1] if geometry else None,
            "pixel_spacing_mm": " x ".join(map(str, geometry.spacing)) if geometry else "",
            "qc_flags": " | ".join(flags),
        })

    times = [time_s for time_s, _ in frames if time_s is not None]
    gaps = np.diff(sorted(set(times)))
    counts = [row["dicom_files"] for row in frame_rows]
    study_row = {
        "patient_id": patient,
        "study_uid": study_uid,
        "mr_series": len(frames),
        "series_with_tt": len(times),
        "duplicate_tt_labels": len(times) - len(set(times)),
        "first_tt_s": min(times, default=None),
        "last_tt_s": max(times, default=None),
        "median_tt_step_s": round(float(np.median(gaps)), 3) if gaps.size else None,
        "max_tt_step_s": round(float(gaps.max()), 3) if gaps.size else None,
        "min_files_per_mr_series": min(counts, default=None),
        "max_files_per_mr_series": max(counts, default=None),
        "ko_series_in_manifest": len(auxiliary),
        "ko_labels_on_disk": " | ".join(sorted(auxiliary_labels)),
        "mr_series_flagged": sum(bool(row["qc_flags"]) for row in frame_rows),
    }
    return study_row, frame_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def slice_index(data_root: Path, row: dict) -> list[tuple[np.ndarray, Path]]:
    """Find slices by physical position in one time volume."""
    slices = []
    folder = data_root / row["series_folder"]
    for path in sorted(folder.glob("*.dcm")):
        ds = dcmread(path, stop_before_pixels=True, specific_tags=HEADER_TAGS)
        if str(ds.SeriesInstanceUID) != row["series_uid"]:
            raise ValueError(f"Series differs from the audit at {path.name}")
        slices.append((np.asarray(ds.ImagePositionPatient, dtype=float), path))
    if len(slices) != int(row["dicom_files"]):
        raise ValueError("DICOM file count changed since the audit. Run again without --report-only.")
    return slices


def load_slice(path: Path) -> np.ndarray:
    from pydicom.pixels import apply_modality_lut

    ds = dcmread(path)
    return np.asarray(apply_modality_lut(ds.pixel_array, ds), dtype=np.float32)


def inspect_workbooks(data_root: Path, output: Path, plt) -> list[dict]:
    from openpyxl import load_workbook

    entries = []
    for path in sorted(data_root.glob("*.xlsx")):
        book = load_workbook(path, read_only=True, data_only=True)
        try:
            for sheet in book.worksheets:
                rows = sheet.values
                headers = tuple(next(rows, ()))
                details = "工作表字段待核实"
                has_aif_plot = False
                if headers[:3] == ("Complete response", None, "Non-complete response"):
                    outcomes = list(rows)
                    complete = sum(bool(row[0]) for row in outcomes)
                    non_complete = sum(bool(row[2]) for row in outcomes)
                    details = f"病理完全响应 {complete} 人，未完全响应 {non_complete} 人。曲线补全不使用这些标签。"
                elif len(headers) >= 2 and "AIF" in str(headers[0]) and "Time Course" in str(headers[1]):
                    points = [
                        (float(row[1]), float(row[0])) for row in rows
                        if isinstance(row[0], (int, float)) and isinstance(row[1], (int, float))
                    ]
                    if points:
                        seconds, concentration = np.asarray(sorted(points)).T
                        fig, ax = plt.subplots(figsize=(8, 3.2), layout="constrained")
                        ax.plot(seconds, concentration, color="#267d96", linewidth=2)
                        ax.set(xlabel="AIF table time (s)", ylabel="Plasma concentration (mM)",
                               title="Population-averaged arterial input function")
                        ax.grid(alpha=0.22)
                        fig.savefig(output / "aif.png", dpi=150)
                        plt.close(fig)
                        has_aif_plot = True
                        details = f"{len(points)} 个时间–浓度样本。群体平均动脉输入函数，用于描述对比剂的输入。"
                entries.append({
                    "file": path.name,
                    "sheet": sheet.title,
                    "columns": " | ".join(str(value) for value in headers if value is not None),
                    "details": details,
                    "has_aif_plot": has_aif_plot,
                })
        finally:
            book.close()
    return entries


def draw_mri_example(data_root: Path, output: Path, frames: list[dict], plt, slice_number: int | None) -> dict:
    from matplotlib.animation import PillowWriter
    from matplotlib.patches import Rectangle

    frames = sorted(frames, key=lambda row: float(row["tt_seconds"]))
    times = np.asarray([float(row["tt_seconds"]) for row in frames])
    reference_slices = slice_index(data_root, frames[0])
    header = dcmread(reference_slices[0][1], stop_before_pixels=True, specific_tags=HEADER_TAGS)
    normal = np.cross(*np.asarray(header.ImageOrientationPatient, dtype=float).reshape(2, 3))
    reference_slices.sort(key=lambda item: item[0] @ normal)
    center = (len(reference_slices) + 1) // 2 if slice_number is None else slice_number
    if not 1 <= center <= len(reference_slices):
        raise ValueError(f"--slice-number must be between 1 and {len(reference_slices)} for this example.")
    position, path = reference_slices[center - 1]
    images = [load_slice(path)]
    for row in frames[1:]:
        candidates = slice_index(data_root, row)
        matches = [path for point, path in candidates if np.allclose(point, position, atol=POSITION_TOL_MM, rtol=0)]
        if len(matches) != 1:
            raise ValueError(f"Could not match the example slice at TT={row['tt_seconds']} s.")
        images.append(load_slice(matches[0]))
    stack = np.stack(images)

    # Keep a single intensity window across time so brightness remains comparable.
    vmin, vmax = np.percentile(stack, [1, 99.5])
    if vmax <= vmin:
        vmax = vmin + 1
    spacing_ratio = float(header.PixelSpacing[0]) / float(header.PixelSpacing[1])
    image_kwargs = dict(cmap="gray", vmin=vmin, vmax=vmax, aspect=spacing_ratio, interpolation="nearest")
    short_id = frames[0]["patient_id"].split("-")[-1]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.5), layout="constrained")
    spatial_indices = np.linspace(0, len(reference_slices) - 1, 5, dtype=int)[1:-1]
    for ax, index in zip(axes[0], spatial_indices):
        ax.imshow(load_slice(reference_slices[index][1]), **image_kwargs)
        ax.set_title(f"TT {times[0]:g} s | Slice {index + 1}/{len(reference_slices)}")
    temporal_indices = [0, len(times) // 3, len(times) - 1]
    for ax, index in zip(axes[1], temporal_indices):
        ax.imshow(stack[index], **image_kwargs)
        ax.set_title(f"Slice {center} | TT {times[index]:g} s")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{short_id} | Top: across slices. Bottom: across time.")
    fig.savefig(output / "mri_montage.png", dpi=150)
    plt.close(fig)

    # Select visible 9x9 patches from the earliest image for teaching, without a lesion mask.
    first = stack[0]
    h, w = first.shape
    threshold = np.percentile(first, 70)
    radius = 4
    candidates = [
        (r, c) for c in range(radius, w - radius, 9) for r in range(radius, h - radius, 9)
        if np.median(first[r-radius:r+radius+1, c-radius:c+radius+1]) > threshold
    ]
    if not candidates:
        candidates = [(h // 2, w // 2)]
    patch_indices = np.unique(np.rint(np.linspace(0, len(candidates) - 1, 5)[1:-1]).astype(int))
    patches = [candidates[index] for index in patch_indices]
    signals = np.column_stack([
        stack[:, r-radius:r+radius+1, c-radius:c+radius+1].mean(axis=(1, 2)) for r, c in patches
    ])
    names = [f"Patch {letter}" for letter in "ABC"[:len(patches)]]
    colors = ["#1989a6", "#e65b51", "#9e6cb1"]
    write_csv(output / "example_signals.csv", [
        {"tt_seconds": t, **{f"{name} (a.u.)": float(value) for name, value in zip(names, values)}}
        for t, values in zip(times, signals)
    ])

    fig, (im_ax, signal_ax) = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    mri = im_ax.imshow(first, **image_kwargs)
    im_ax.axis("off")
    for index, ((r, c), color, name) in enumerate(zip(patches, colors, names)):
        im_ax.add_patch(Rectangle((c-radius-0.5, r-radius-0.5), 9, 9, fill=False, edgecolor=color, lw=2))
        im_ax.text(c + 6, r, name[-1], color=color, weight="bold", va="center",
                   bbox=dict(facecolor="white", edgecolor="none", pad=1))
        signal_ax.plot(times, signals[:, index], color=color, label=name, linewidth=1.7, marker=".")
    line = signal_ax.axvline(times[0], color="#555555", linestyle="--", linewidth=1)
    signal_ax.set(xlabel="TT from DICOM series label (s)", ylabel="Mean MR signal (a.u.)",
                  title="Measured signals in the marked patches")
    signal_ax.grid(alpha=0.22)
    signal_ax.legend(frameon=False)
    im_ax.set_title(f"{short_id} | Slice {center} | TT {times[0]:g} s")
    fig.savefig(output / "signal_curves.png", dpi=150)
    fig.set_layout_engine("none")
    writer = PillowWriter(fps=5)
    with writer.saving(fig, str(output / "dce_playback.gif"), dpi=95):
        for t, pixels in zip(times, stack):
            mri.set_data(pixels)
            im_ax.set_title(f"{short_id} | Slice {center} | TT {t:g} s")
            line.set_xdata([t, t])
            writer.grab_frame()
    plt.close(fig)
    return {"patient": short_id, "slice": center, "slices": len(reference_slices), "frames": len(times),
            "shape": tuple(first.shape), "positions": patches}


def visit_label(study: dict) -> str:
    match = re.search(r"\bVisit\s+([12])\b", study["ko_labels_on_disk"], re.IGNORECASE)
    return match.group(1) if match else ""


def html_table(headers, rows) -> str:
    header_html = "".join(f"<th scope='col'>{escape(str(value))}</th>" for value in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{escape(str(value)) if value not in (None, '') else '—'}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return f"<div class='table-wrap'><table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table></div>"


def build_learning_report(data_root: Path, output: Path, studies: list[dict], frames: list[dict], args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    patient = args.example_patient.upper() if args.example_patient else None
    suitable = [
        study for study in studies
        if visit_label(study) == args.example_visit
        and (patient is None or study["patient_id"].split("-")[-1].upper() == patient)
        and int(study["series_with_tt"]) >= 3
        and int(study["mr_series_flagged"]) == 0
        and int(study["duplicate_tt_labels"]) == 0
    ]
    if not suitable:
        raise ValueError("No matching example with usable timing and geometry. See studies.csv.")
    study = min(suitable, key=lambda row: (-int(row["series_with_tt"]), row["patient_id"], row["study_uid"]))
    example_frames = [
        row for row in frames
        if row["patient_id"] == study["patient_id"] and row["study_uid"] == study["study_uid"]
    ]
    print(f"Decoding one spatial slice across time for {study['patient_id']}, Visit {args.example_visit}…", flush=True)
    example = draw_mri_example(data_root, output, example_frames, plt, args.slice_number)
    workbooks = inspect_workbooks(data_root, output, plt)
    workbook_table = html_table(
        ["文件", "工作表", "字段", "对项目有什么用"],
        [(row["file"], row["sheet"], row["columns"], row["details"]) for row in workbooks],
    ) if workbooks else "<p>数据根目录没有找到 XLSX 文件。</p>"
    aif_figure = """<figure><img src="aif.png" alt="群体平均 AIF 曲线">
      <figcaption>这张图的纵轴是血浆对比剂浓度，单位 mM，横轴来自 AIF 表。图中的时间轴未与 MRI 的 TT 对齐。</figcaption>
      </figure>""" if any(row["has_aif_plot"] for row in workbooks) else ""
    study_table = html_table(
        ["患者", "访视", "MR 时间帧", "每帧 DICOM 数（最少–最多）", "TT 间隔中位数 (s)", "有标记的帧"],
        [(row["patient_id"].split("-")[-1], visit_label(row), row["series_with_tt"],
          f"{row['min_files_per_mr_series']}–{row['max_files_per_mr_series']}", row["median_tt_step_s"],
          row["mr_series_flagged"]) for row in sorted(studies, key=lambda row: (row["patient_id"], visit_label(row)))],
    )
    min_frames = min(int(row["series_with_tt"]) for row in studies)
    max_frames = max(int(row["series_with_tt"]) for row in studies)
    patient_count = len({row["patient_id"] for row in studies})
    ko_count = sum(bool(row["ko_labels_on_disk"]) for row in studies)
    loose = sum(
        path.is_file() for path in data_root.rglob("*") if path.name.lower().endswith((".nii", ".nii.gz", ".mat"))
    )
    height, width = example["shape"]
    coordinates = "；".join(f"{letter} = ({r}, {c})" for letter, (r, c) in zip("ABC", example["positions"]))
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>00 · QIN Breast DCE-MRI 数据导览</title>
<style>
body {{ max-width: 1000px; margin: 32px auto; padding: 0 20px; font: 16px/1.65 system-ui, 'Microsoft YaHei', sans-serif; color: #17304a; background: #f5f8fb; }}
section {{ background: white; border: 1px solid #dae4ee; border-radius: 12px; margin: 20px 0; padding: 16px 22px; }}
h1, h2 {{ line-height: 1.25; }} a {{ color: #145f78; }}
.table-wrap {{ overflow-x: auto; }} table {{ width: 100%; border-collapse: collapse; font-size: .92em; }}
th, td {{ text-align: left; padding: 6px 9px; border-bottom: 1px solid #e0e8ee; }} th {{ background: #eaf2f8; }}
figure {{ margin: 14px 0 22px; }} img {{ display: block; width: 100%; height: auto; }}
figcaption {{ margin-top: 6px; color: #354b5f; }}
</style></head><body>
<h1>QIN Breast DCE-MRI 数据导览</h1>
<p>一次 DCE-MRI 检查连续采集多个三维体积，用来观察注入对比剂后 MRI 信号如何随位置和时间变化。</p>
<section><h2>1. 一条数据有多大？</h2>
<p><strong>{patient_count} 位患者 → {len(studies)} 次检查 → 每次检查 {min_frames}–{max_frames} 个带 TT 的时间帧。</strong>
一个时间帧是一份 3D 体积，由多张二维 DICOM 切片堆叠。空间轴是切片、行、列，加上时间轴后是一份 4D 序列。</p>
<p>同一患者的 Visit 1 和 Visit 2 来自两次不同检查。这个项目要补的是单次检查内部的动态时间点。
TT 来自 DICOM 序列名称，报告沿用该标签的秒数。</p>
{study_table}
<p><a href="studies.csv">逐次检查的完整表</a> · <a href="mr_series.csv">逐帧时间、几何和 QC 表</a></p></section>
<section><h2>2. 打开一份真实序列</h2>
<p>示例是 {escape(example['patient'])} · Visit {args.example_visit}，每帧包含 {example['slices']} 张 {height} × {width} 切片。
图像采用统一灰度范围，可以直接比较不同帧的亮度。</p>
<figure><img src="mri_montage.png" alt="同一时间的三张切片与同一切片的三个时刻">
<figcaption>上排固定时间，沿空间看三张切片。下排固定切片，沿时间看早、中、晚三个时刻。</figcaption></figure>
<figure><img src="dce_playback.gif" alt="MRI 切片与信号曲线同步播放" loading="lazy">
<figcaption>竖虚线指向当前播放帧。动画使用加速播放，各帧实际时间读图上的 TT 标签。</figcaption></figure>
<figure><img src="signal_curves.png" alt="彩色采样框与对应的平均 MRI 信号曲线" loading="lazy">
<figcaption>每条曲线是在同一张切片上，对同色 9 × 9 小框内的像素逐帧求平均。框由首帧自动选取，仅供演示。
框中心的零基坐标为（行，列）{escape(coordinates)}。信号单位为 a.u.，没有做配准或浓度换算。
<a href="example_signals.csv">下载曲线数值</a></figcaption></figure>
</section>
<section><h2>3. 图像之外还有什么？</h2>
{workbook_table}{aif_figure}
<p>在 {ko_count} 次检查中找到了列在清单里的 DICOM-KO 辅助对象。本地另发现 {loose} 个独立的 NIfTI 或 MATLAB 文件。
当前按文件名和清单识别辅助对象，KO 附件和病灶 ROI 仍待核实。</p></section>
<p>来源 <a href="https://doi.org/10.7937/K9/TCIA.2014.A2N1IXOX">TCIA QIN Breast DCE-MRI</a>，
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC3998693/">原始研究的采集与数据说明</a>。</p>
</body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    print(f"Learning report: {output / 'index.html'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="QIN download directory")
    parser.add_argument("--report-only", action="store_true", help="Reuse existing audit CSV files")
    parser.add_argument("--example-patient", help="Short patient ID for the preview, e.g. BC05")
    parser.add_argument("--example-visit", choices=("1", "2"), default="1", help="Visit to preview")
    parser.add_argument("--slice-number", type=int, help="One-based spatial slice number (default: middle slice)")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    manifest = data_root / "metadata" / "metadata.csv"
    output = (PROJECT_ROOT / "outputs" / "00_data_audit").resolve()
    if output.is_relative_to(data_root):
        parser.error("The audit output must be outside the raw data directory.")
    if not manifest.is_file():
        parser.error(f"Manifest not found: {manifest}")
    if args.report_only:
        if not all((output / name).is_file() for name in ("studies.csv", "mr_series.csv")):
            parser.error("Audit CSV files not found. Run again without --report-only.")
        build_learning_report(data_root, output, read_csv(output / "studies.csv"),
                              read_csv(output / "mr_series.csv"), args)
        return

    required_columns = {
        "PatientID", "StudyInstanceUID", "SeriesInstanceUID", "SeriesNumber",
        "SeriesDescription", "Modality", "S5cmdManifestPath",
    }
    grouped = defaultdict(list)
    with manifest.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            parser.error(f"Missing manifest columns: {', '.join(sorted(missing))}")
        for row in reader:
            grouped[(row["PatientID"], row["StudyInstanceUID"])].append(row)
    if not any(row["Modality"] == "MR" for rows in grouped.values() for row in rows):
        parser.error("The manifest contains no MR series.")

    studies = []
    frames = []
    for index, ((patient, _), rows) in enumerate(sorted(grouped.items()), start=1):
        print(f"[{index}/{len(grouped)}] Reading {patient} DICOM headers", flush=True)
        study, study_frames = audit_study(data_root, rows)
        studies.append(study)
        frames.extend(study_frames)

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "studies.csv", studies)
    write_csv(output / "mr_series.csv", frames)

    counts = Counter(row["Modality"] for rows in grouped.values() for row in rows)
    print(f"\nPatients: {len({patient for patient, _ in grouped})}  Studies: {len(studies)}")
    print(f"Manifest series by modality: {dict(sorted(counts.items()))}")
    print(f"Flagged MR series: {sum(bool(row['qc_flags']) for row in frames)}")
    print(f"Reports: {output}")
    build_learning_report(data_root, output, studies, frames, args)


if __name__ == "__main__":
    main()
