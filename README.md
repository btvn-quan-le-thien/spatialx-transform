# spatialx-transform

Spatial image transformation library for aligning and warping large-scale microscopy images. Supports Affine, TPS, and composed transforms with zarr-based streaming I/O for out-of-core processing.

## Installation

```bash
uv add spatialx-transform
```

## Quick Start

```python
import json
from spatialx_transform.transforms import Transformation
from spatialx_transform.warp import warp_transform

# 1. Load alignment from JSON
with open("alignment.json") as f:
    aln_data = json.load(f)
align_tf = Transformation.model_validate(aln_data["alignment"]["transformation"])

# 2. Preflight — check output shape without writing
preflight = warp_transform(
    input_dir="input.zarr",
    output_dir="output_dir",
    tf=align_tf,
    d=(50, 50),
    scale=(0.5, 0.5),
    preflight=True,
)
print(f"Output shape: {preflight.img_shape}, offset: {preflight.offset}")

# 3. Run the warp — writes per-channel zarr arrays to output_dir
result = warp_transform(
    input_dir="input.zarr",
    output_dir="output_dir",
    tf=align_tf,
    d=(50, 50),
    scale=(0.5, 0.5),
    chunk_size=(50000, 50000),
)

# 4. Read output
import numpy as np
channel_0 = np.asarray(result.img_zarr_list[0])
```

## `warp_transform()` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_dir` | `str` | *(required)* | Path to input zarr store. Shape `[C, H, W]` (3D) or `[H, W]` (2D). |
| `output_dir` | `str` | *(required)* | Output directory. Writes `img_output_channel{c}.zarr` (3D) or `img_output.zarr` (2D). |
| `tf` | `Transformation` | *(required)* | Transform to apply. Supports `Affine`, `TPS`, `Composed`, `Identity`. Can be loaded from JSON via `Transformation.model_validate(data)`. |
| `d` | `tuple[int, int]` | `(1, 1)` | Triangle mesh grid step `(dx, dy)`. Smaller = finer mesh = more accurate but slower. |
| `chunk_size` | `tuple[int, int]` | `(1, 1)` | Source chunk size `(H, W)`. Controls RAM usage. Larger = faster but more memory. |
| `scale` | `tuple[float, float]` | `(1.0, 1.0)` | Output resolution factor. `<1` downsamples, `>1` upsamples. |
| `preflight` | `bool` | `False` | If `True`, computes output shape only without writing data. |

## Transformations

```python
from spatialx_transform.transforms import Affine, TPS, Identity
from spatialx_transform.params import AffineParams

# Affine: p' = A @ p + b
tf = Affine(params=AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[5.0, 5.0]))

# Or load from JSON (supports affine, tps, composed, identity)
tf = Transformation.model_validate({
    "transformation_type": "affine",
    "params": {"A": [[1.0, 0.0], [0.0, 1.0]], "b": [5.0, 5.0]}
})
```

## Recommended Parameters

| Image size (H × W) | `d` (dx, dy) | `chunk_size` | `scale` |
|---|---|---|---|
| 100K x 30k | `(50, 50)` | `(45000, 15000)` | `(1.0, 1.0)` |
---

## Examples

See `demo.ipynb` for a full walkthrough.
