# TreeLearn Setup & Quickstart

This repository has been configured with complete **NVIDIA CUDA acceleration** on your NVIDIA GeForce RTX 3090.

---

## Environment Overview

- **Repository**: `arunbiju969/TreeLearn`
- **Location**: `C:\Users\w0491597\Documents\Projects\TreeLearn`
- **Python Version**: CPython 3.10 (managed in `.venv`)
- **PyTorch**: `2.6.0+cu124` (with CUDA 12.4 support)
- **Sparse 3D Convolution**: `spconv-cu124==2.3.8` (CUDA accelerated)
- **Key Dependencies**:
  - `numpy==1.26.4` (required for C extension compatibility with `jakteristics` and `open3d`)
  - `open3d==0.17.0`
  - `jakteristics==0.6.2`
  - `hdbscan`, `scikit-learn`, `scipy`, `pandas`
  - `laspy[lazrs]==2.7.0`
  - `geopandas`, `shapely`, `alphashape`
  - `timm`, `munch`, `pyyaml`, `tensorboard`, `tensorboardX`

---

## Pre-Downloaded Assets

The required model weights and benchmark point cloud have already been downloaded:
1. **Model Weights** (`data/model_weights/`):
   - `model_weights_20241213.pth` (standard model, trained without outlier filtering)
   - `model_weights_with_small_20241213.pth` (detects trees < 10 m tall)
2. **Benchmark Forest Point Cloud** (`data/pipeline/L1W/forest/`):
   - `L1W.laz` (342 MB benchmark point cloud)

---

## Running the Pipeline

### 1. Using Quick Launchers
- **Batch**: Double-click `run_pipeline.bat`
- **PowerShell**:
  ```powershell
  .\run_pipeline.ps1
  ```

### 2. Manual Command Line
```powershell
.\.venv\Scripts\python.exe tools/pipeline/pipeline.py --config configs/pipeline/pipeline.yaml
```

### Configuration Notes (`configs/pipeline/pipeline.yaml`)
- `forest_path`: Path to point cloud (`.laz`, `.las`, `.npy`, `.npz`, or `.txt`). Currently set to `data/pipeline/L1W/forest/L1W.laz`.
- `tile_generation`: Set to `True` for the first run on a new point cloud to generate tiles. Set to `False` on subsequent runs to reuse cached tiles.
- `pretrain`: Path to the `.pth` model weights.
- `results_dir`: Output directory (`data/pipeline/L1W/results/` by default).

---

## Verifying CUDA & spconv

Run this one-liner to verify GPU and sparse convolution:
```powershell
.\.venv\Scripts\python.exe -c "import torch, spconv.pytorch as spconv; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0)); f=torch.randn(10,16,device='cuda'); i=torch.zeros((10,4),dtype=torch.int32,device='cuda'); t=spconv.SparseConvTensor(f,i,[5,5,5],1); print('spconv CUDA OK:', spconv.SubMConv3d(16,32,3,indice_key='subm').cuda()(t).features.shape)"
```
