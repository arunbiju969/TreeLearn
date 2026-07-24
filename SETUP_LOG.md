# TreeLearn Setup Log

Setup of the [TreeLearn](https://github.com/ecker-lab/TreeLearn) segmentation pipeline on this machine, following the [segmentation pipeline guide](docs/segmentation_pipeline.md). Completed and verified on 2026-07-23/24.

## Hardware / Environment

- **GPU:** NVIDIA GeForce RTX 5090 Laptop (Blackwell, compute capability sm_120)
- **RAM:** 64 GB
- **OS:** Windows
- **Conda:** miniforge3 at `C:\Users\arunb\miniforge3` (not on PATH)
- **Env:** `TreeLearn` (Python 3.10) at `C:\Users\arunb\miniforge3\envs\TreeLearn`

## Steps performed

1. Cloned `https://github.com/ecker-lab/TreeLearn.git` into `d:\Projects\TreeLearn`.
2. Created conda env: `conda create -n TreeLearn python=3.10`.
3. Installed PyTorch + dependencies (see deviations below).
4. Installed package: `pip install -e .`
5. Downloaded benchmark data and model weights:
   - `python tree_learn/util/download.py --dataset_name benchmark_dataset --root_folder data/pipeline/L1W/forest` (342 MB)
   - `python tree_learn/util/download.py --dataset_name model_weights_20241213 --root_folder data/model_weights`
   - `python tree_learn/util/download.py --dataset_name model_weights_with_small_20241213 --root_folder data/model_weights`
6. Ran the pipeline end-to-end on L1W and verified results.

## Deviations from the stock setup

The stock `setup/setup.sh` pins CUDA 11.8 (pytorch 2.0.0 + spconv-cu118), which **does not support sm_120**. Installed instead:

| Component | Stock | Installed |
|-----------|-------|-----------|
| PyTorch | 2.0.0 (cu118) | 2.11.0+cu128 |
| spconv | spconv-cu118 | spconv-cu126 2.3.8 |
| numpy | 1.24.4 | 1.26.4 |

Additional packages installed: `hdbscan` (default grouping), `tensorboard`, `tensorboardX`.

- numpy had to stay on 1.x because `jakteristics`' compiled C extension fails on numpy 2.x with `ValueError: numpy.dtype size changed`.
- Local requirements file: `setup/requirements_local.txt`.

## Code patches required on this machine

### 1. Fix native crash in tile generation — `tree_learn/util/data_preparation.py`

The "center chunks" step moved every chunk to the GPU and back just to subtract a center offset:

```python
valid_chunks[i] = (torch.from_numpy(valid_chunks[i]).cuda() - torch.from_numpy(center).cuda()).cpu().numpy()
```

On this GPU/driver combo this caused a **Windows access violation** (`0xC0000005`, exit code `-1073741819`) with no Python traceback (diagnosed via `python -X faulthandler`). Replaced with an in-place CPU subtraction:

```python
valid_chunks[i] -= center
```

The in-place form also avoids a large temporary copy (an earlier non-in-place version failed with `ArrayMemoryError`).

### 2. Memory-efficient ensembling — `tree_learn/util/pipeline.py` (`ensemble()`)

The original built 8 separate pandas DataFrames, `pd.concat`-ed them into one huge frame (82M rows), then `df.round(...)` (another full copy). This failed with `ArrayMemoryError: Unable to allocate 1.84 GiB` on 64 GB RAM (the docs note L1W peaks near 100 GB).

Rewrote it to:
- build a **single combined float32 numpy array** and one DataFrame (`copy=False`),
- free the large input arrays immediately after copying,
- round coordinates **in place** per-column,
- then `groupby(['x','y','z']).mean()`.

This roughly halves peak memory of the step. Behavior is unchanged.

## Running the pipeline

```powershell
cd d:\Projects\TreeLearn
C:\Users\arunb\miniforge3\envs\TreeLearn\python.exe tools/pipeline/pipeline.py --config configs/pipeline/pipeline.yaml
```

- Config: `configs/pipeline/pipeline.yaml` (`forest_path`, `pretrain`, `tile_generation`, etc.)
- On re-runs for the same point cloud, set `tile_generation: False` (tiles are cached in `data/pipeline/L1W/tiles/`).

## Verified result (L1W benchmark)

Full run completed (exit code 0):

- Tile generation: 842 tiles (~12 min first run)
- GPU inference: 1221 batches (~5 min)
- Ensembling: ~1 min
- Instance grouping: ~13 min
- **305 trees segmented**: 219 `completely_inside`, 83 `trunk_base_inside`, 3 `trunk_base_outside`

Output written to `data/pipeline/L1W/results/`:
- `full_forest/L1W.laz` (333 MB, full segmented forest)
- `individual_trees/` (per-tree `.las` files + `non_trees.las`)
- `pointwise_results/` (`pointwise_results.npz`, `cluster_coords.laz`)

## Segmenting a custom forest

1. Place the point cloud (`.las`/`.laz`/`.npy`/`.npz`/space-delimited `.txt`) in `data/pipeline/L1W/forest/`.
2. Set `forest_path` in `configs/pipeline/pipeline.yaml` to the file; set `tile_generation: True`.
3. Run the command above.

Data requirements: meter scale, ≥ 1 point per (0.1 m)^3, well-resolved trunks, terrain + low vegetation included, only rough noise pre-filtering. Keep a ~13.5 m buffer around the area of interest. To also detect trees < 10 m tall, set `pretrain` to `data/model_weights/model_weights_with_small_20241213.pth`.

Note: 64 GB RAM is below the ~100 GB peak cited in the docs for L1W; the ensemble patch keeps it working, but very large custom clouds may still be memory-bound.
