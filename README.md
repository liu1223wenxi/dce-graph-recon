# Reconstructing Breast DCE-MRI Curves with a Graph Neural Network

Can neighboring image regions help recover missing parts of a contrast-enhancement curve? This project hides measurements from real breast DCE-MRI scans and compares reconstructions against the original measurements. A graph neural network (GNN) reduced local-gap error by **6.1%** relative to linear interpolation. A similarly sized model using only the target region's time series matched that gain, so this benchmark found little added predictive value from the graph.

## How the model uses the graph

1. **Build one spatial graph per visit.** Take one central 2D slice at the same physical location across the retained frames. Use the first retained image to define the foreground and divide it into SLIC superpixels. Each superpixel is a node with a mean relative-enhancement curve. Regions that share a boundary are connected.
2. **Interpolate in time.** Start with linear interpolation between observed time points on the target node's own curve. Test two patterns: alternating missing whole frames and a consecutive gap affecting a compact group of regions. In the local gap, some adjacent regions still have actual measurements at the missing times.
3. **Learn a correction.** For each incoming edge, an MLP uses the target and neighbor features and their spatial distance to predict a weight and a signed correction. The GNN normalizes the weights across the target's immediate neighbors and adds their weighted corrections to the linear estimate at each missing point. The input features distinguish measured signals from interpolated ones. Observed points are preserved. A time-only residual MLP with a similar parameter count tests how much improvement is possible using the target curve alone.

## Results

The benchmark uses 20 visits from 10 patients. Five patient-grouped folds keep each patient's visits together and hold every patient out for testing once. In each fold, six patients train the model, two select the checkpoint and two are held out for testing. All methods use the same hidden points. MAE is averaged over visits within each patient and then across patients. Values are percentage points of relative signal enhancement, and lower is better.

| Missing pattern | Linear interpolation | Time-only MLP | GNN | GNN improvement vs. linear | Patients improved vs. linear |
| --- | ---: | ---: | ---: | ---: | ---: |
| Alternating whole frames | 7.079 | 7.060 | **7.056** | 0.023 pp (0.32%) | 6/10 |
| Local consecutive gap | 15.392 | **14.436** | 14.453 | 0.939 pp (6.10%) | 10/10 |

- **Did the GNN use observed neighbors?** On eligible local-gap points with observed neighbors, replacing their same-time measurements with temporal interpolants increased GNN MAE from 13.431 to 13.495 pp. Error rose for 9 of 10 patients on that subset. The model uses those observations, while the full local-gap comparison shows that a time-only model can reach the same accuracy.
- **Did it recover the curve peak more accurately?** For regions whose measured peak fell in the local gap, peak-height MAE was 12.257 pp for linear interpolation, 12.240 pp for the time-only MLP and 12.216 pp for the GNN. The improvement over linear interpolation was 0.041 pp on this exploratory endpoint.

The evaluation uses artificial gaps on a small cohort and one 2D slice per visit. Enhancement is relative to the first retained image, and peak errors are measured on superpixel curves. Clinical lesion measurements, real acquisition failures and performance on external data remain untested.

## Reproduce the experiment

Use Python 3.11. These commands are for Windows Command Prompt and run from the repository root.

```bat
conda create -n dce-graph-recon python=3.11
conda activate dce-graph-recon
python -m pip install numpy scipy scikit-image pydicom matplotlib openpyxl
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Download the QIN Breast DCE-MRI collection separately. The data audit expects the following local layout:

```text
<data-root>/
  metadata/
    metadata.csv
  qin_breast_dce_mri/
    .../
      <series folder>/
        *.dcm
```

The CSV must contain `PatientID`, `StudyInstanceUID`, `SeriesInstanceUID`, `SeriesNumber`, `SeriesDescription`, `Modality` and `S5cmdManifestPath`. Each `S5cmdManifestPath` must include the `qin_breast_dce_mri` folder and identify the series directory beneath it. Keep the MR frames and the DICOM-KO series labeled `NIfTI Visit 1` or `NIfTI Visit 2` in the input. Stage 01 uses those labels to identify visits. A download with a different folder layout or manifest format needs to be mapped to this input format.

```bat
set "DATA_ROOT=D:\data\QIN Breast DCE-MRI"
python scripts\00_inspect_data.py --data-root "%DATA_ROOT%"
python scripts\01_prepare_data.py --data-root "%DATA_ROOT%"
python scripts\02_build_graphs.py
python scripts\03_run_baselines.py
python scripts\04_train_gnn.py --device cpu
python scripts\05_evaluate.py --device cpu
python scripts\06_ablate_neighbors.py --device cpu
python scripts\07_evaluate_peaks.py --device cpu
```

Set `DATA_ROOT` to your download location. The reported run used the default training settings on CUDA. To use CUDA, install a [compatible PyTorch build](https://pytorch.org/get-started/locally/) in place of the CPU build and use `--device cuda` in stages 04–07.

## Read the generated reports

Each stage writes an `index.html` under its report directory. The reports include example images and plots with explanations in Chinese.

| Stage | Directory under `outputs/` | What to look for |
| --- | --- | --- |
| 00–01 | `00_data_audit/`, `01_prepared/` | Acquisition times, image geometry, example slices and enhancement curves |
| 02–03 | `02_graphs/`, `03_baselines/` | Superpixels and graph edges, the two masks, and comparisons among linear interpolation, PCHIP and fixed graph smoothing |
| 04–05 | `04_training/`, `05_evaluation/` | Patient splits, training curves, held-out errors and example neighbor contributions |
| 06 | `06_ablation/` | The time-only comparison and the effect of replacing measured neighbor values |
| 07 | `07_peaks/` | Peak-height error for regions affected by the local gap |

Prepared arrays and graph samples are generated under `<data-root>/derived/dce-graph-recon/`. Reports, result CSVs and checkpoints are generated locally under `outputs/`.

## Dataset credit

Huang, W., Tudorica, A., Chui, S., Kemmer, K., Naik, A., Troxell, M., Oh, K., Roy, N., Afzal, A., & Holtorf, M. (2014). *QIN Breast DCE-MRI* (Version 2) [Data set]. The Cancer Imaging Archive. [DOI 10.7937/K9/TCIA.2014.A2N1IXOX](https://doi.org/10.7937/K9/TCIA.2014.A2N1IXOX).
