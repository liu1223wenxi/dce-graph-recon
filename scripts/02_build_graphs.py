"""Build one region adjacency graph per prepared DCE slice."""

import argparse
from importlib import import_module

import numpy as np
from matplotlib.collections import LineCollection
from skimage.segmentation import find_boundaries, slic

prepare = import_module("01_prepare_data")
audit, plt = prepare.audit, prepare.plt
GRAPH_DIR = prepare.OUTPUTS / "02_graphs"


def build_graph(data, segments: int) -> dict:
    reference = data["signal"][0]
    foreground = data["foreground"]
    spacing = data["spacing_mm"]
    p_low, p_high = np.percentile(reference[foreground], [1, 99])
    guide = np.clip((reference - p_low) / max(1e-6, p_high - p_low), 0, 1)
    labels = slic(guide, n_segments=segments, compactness=0.2, sigma=1, spacing=spacing,
                  mask=foreground, start_label=1, channel_axis=None)
    labels[~foreground] = 0
    original = np.unique(labels[labels > 0])
    labels = np.searchsorted(np.r_[0, original], labels)
    if len(original) < 2:
        raise ValueError("SLIC returned fewer than two nodes.")
    valid = labels > 0
    # Label 0 is background. Graph node i corresponds to image label i+1.
    node_ids = labels[valid] - 1
    count = np.bincount(node_ids)
    rr, cc = np.nonzero(valid)
    centroids = np.column_stack([
        np.bincount(node_ids, weights=rr) / count,
        np.bincount(node_ids, weights=cc) / count,
    ])
    reference_mean = np.bincount(node_ids, weights=reference[valid]) / count
    signals = np.column_stack([
        np.bincount(node_ids, weights=frame[valid]) / count for frame in data["relative_pct"]
    ]).astype(np.float32)

    # Use shared sides only. Diagonal touch does not make an edge.
    left = np.column_stack((labels[:, :-1].ravel(), labels[:, 1:].ravel()))
    above = np.column_stack((labels[:-1, :].ravel(), labels[1:, :].ravel()))
    sides = np.vstack((left, above))
    touching = sides[(sides[:, 0] != sides[:, 1]) & (sides.min(axis=1) > 0)] - 1
    edges, shared_sides = np.unique(np.sort(touching, axis=1), axis=0, return_counts=True)
    distance_mm = np.linalg.norm((centroids[edges[:, 0]] - centroids[edges[:, 1]]) * spacing, axis=1)
    return dict(labels=labels.astype(np.int32), signals_pct=signals, times_s=data["times_s"],
                reference=reference, centroids_rc=centroids, centroids_mm=centroids * spacing,
                node_pixels=count, node_reference_mean=reference_mean,
                edges=edges.astype(np.int32), edge_distance_mm=distance_mm, edge_shared_sides=shared_sides)


def save_preview(sample: str, graph: dict) -> str:
    labels = graph["labels"]
    centers = graph["centroids_rc"][:, ::-1]  # Matplotlib expects x=column, y=row.
    reference = graph["reference"]
    limits = np.percentile(reference[labels > 0], [1, 99])
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), layout="constrained")
    for ax in axes[:2]:
        ax.imshow(reference, cmap="gray", vmin=limits[0], vmax=limits[1])
        ax.axis("off")
    boundaries = find_boundaries(labels, mode="inner", background=0) & (labels > 0)
    axes[0].imshow(np.ma.masked_where(~boundaries, boundaries), cmap="cool", vmin=0, vmax=1, alpha=0.75)
    axes[0].set_title("Superpixel boundaries")
    axes[1].add_collection(LineCollection(centers[graph["edges"]], colors="#59c2d9", linewidths=0.65, alpha=0.7))
    axes[1].scatter(*centers.T, s=5, c="#e3f5f7")
    axes[1].set_title("Spatial adjacency graph")
    nodes = np.unique(np.linspace(0, len(centers) - 1, 5, dtype=int)[1:-1])
    colors = ("#2088a6", "#c55b58", "#926cb2")
    for node, color in zip(nodes, colors):
        axes[1].scatter(*centers[node], s=30, c=color, edgecolors="white", linewidths=0.5)
        axes[1].annotate(str(node), centers[node], xytext=(4, 4), textcoords="offset points",
                         color="white", fontsize=8)
        axes[2].plot(graph["times_s"], graph["signals_pct"][node], label=f"Node {node}", color=color)
    axes[2].set(title="Regional signal curves", xlabel="TT (s)", ylabel="Change vs. reference (%)")
    axes[2].grid(alpha=0.2)
    axes[2].legend(frameon=False)
    fig.suptitle(sample)
    fig.savefig(GRAPH_DIR / "graph_preview.png", dpi=150)
    plt.close(fig)
    return "、".join(str(node) for node in nodes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=int, default=160, help="Approximate target number of superpixels")
    args = parser.parse_args()
    if args.segments < 4:
        parser.error("--segments must be at least 4.")
    source = prepare.PREPARED_DIR / "manifest.csv"
    if not source.is_file():
        parser.error("Run 01_prepare_data.py first.")
    samples = audit.read_csv(source)
    if not samples:
        parser.error("The preparation manifest is empty.")

    # Keep graph samples beside the prepared samples' stage on the data drive.
    prepared_paths = [prepare.PREPARED_DIR / row["file"] for row in samples]
    prepared_dirs = {path.parent for path in prepared_paths}
    if len(prepared_dirs) != 1:
        parser.error("Expected one prepared sample directory in the 01 manifest.")
    prepared_samples = prepared_paths[0].parent
    if prepared_samples.name != "samples" or prepared_samples.parent.name != "01_prepared":
        parser.error("Expected 01_prepared/samples in the paths from the 01 manifest.")
    graph_samples = prepared_samples.parent.with_name("02_graphs") / "samples"
    graph_samples.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for index, (row, sample_path) in enumerate(zip(samples, prepared_paths)):
        print(f"Building graph for {row['sample']}", flush=True)
        with np.load(sample_path, allow_pickle=False) as data:
            graph = build_graph(data, args.segments)
        graph_path = graph_samples / f"{row['sample']}.npz"
        np.savez_compressed(graph_path, **graph)
        nodes, frames = graph["signals_pct"].shape
        edges = len(graph["edges"])
        degree = np.bincount(graph["edges"].ravel(), minlength=nodes)
        summary.append(dict(sample=row["sample"], patient=row["patient"], visit=row["visit"], nodes=nodes,
                            edges=edges, frames=frames, mean_degree=round(2 * edges / nodes, 2),
                            isolated_nodes=int(np.count_nonzero(degree == 0)), file=graph_path.as_posix()))
        if index == 0:
            example_nodes = save_preview(row["sample"], graph)
    audit.write_csv(GRAPH_DIR / "manifest.csv", summary)
    table = audit.html_table(["样本", "节点", "无向边", "时间帧", "平均度", "孤立节点"], [
        (r["sample"], r["nodes"], r["edges"], r["frames"], r["mean_degree"], r["isolated_nodes"]) for r in summary
    ])
    example = summary[0]
    prepare.write_report(GRAPH_DIR, "02 · 把 MRI 切片变成空间图", f"""
<section><h2>02 做了什么</h2>
<p>01 给了我们一张随时间变化的 MRI 切片。02 把切片划成小区域，每个区域作为一个<strong>节点</strong>。每个节点有自己的信号曲线，也知道哪些区域紧挨着它。03 会利用这些邻居关系做一次补全实验。</p>
<p>图中示例 <strong>{example["sample"]}</strong> 有 <strong>{example["nodes"]} 个节点、{example["edges"]} 条边、{example["frames"]} 个时间点</strong>。可以把它想成一张 {example["nodes"]} × {example["frames"]} 的表，每行一个区域，每列一个时间点，再加一份「哪些行互为邻居」的名单。</p></section>
<section><h2>从左到右读图</h2>
<figure><img src="graph_preview.png" alt="左图是 MRI 分区，中图是各分区之间的邻接关系，右图是标记区域的信号曲线"></figure>
<ol>
<li><strong>左图看区域。</strong>灰底是检查的首帧。紫色线圈出由 SLIC 划分的像素小区域。划分只用首帧的外观和像素位置，一次划好后对整段序列复用。</li>
<li><strong>中图看邻居。</strong>每个点放在区域中心。两个区域如果有像素沿边相接，就把两个中心画一条线。连线表达邻接，MRI 亮度看灰底。图中标记的 {example_nodes} 是区域编号。</li>
<li><strong>右图看区域怎样随时间变化。</strong>用相同颜色和编号找回中图的位置。蓝线、红线、紫线分别来自对应区域。每个时间点的纵轴值是该区域内像素相对首帧的百分比变化取平均。首帧因此从 0% 起步。横轴 TT 是扫描序列标签上的时间（秒）。</li>
</ol>
<p>可以试着跟着一条线读。例如纵轴到 +20%，表示这个区域内像素的相对变化平均约为 +20%。如果某个点被藏起来，左、右相邻的时间点属于<strong>同一区域的时间信息</strong>，中图和它相连的点则提供<strong>其他区域的空间信息</strong>。03 比较这两种信息怎么用。</p></section>
<section>{table}<p><a href="manifest.csv">图清单和 NPZ 路径</a></p></section>""")
    print(f"Graph samples: {graph_samples}")
    print(f"Report: {GRAPH_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
