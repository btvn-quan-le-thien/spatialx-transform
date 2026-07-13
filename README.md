# spatialx-transform

Spatial image transformation library for aligning and warping large-scale microscopy images. Supports Affine, TPS, and composed transforms with zarr-based streaming I/O for out-of-core processing.

## Installation

### From PyPI (uv)

```bash
uv add spatialx-transform
```

### From git (pip)

```bash
# latest from main
pip install git+https://github.com/btvn-quan-le-thien/spatialx-transform.git

# pinned to a release tag
pip install git+https://github.com/btvn-quan-le-thien/spatialx-transform.git@v0.3.1
```

### From a built wheel

```bash
# 1. Build the wheel
uv build

# 2. Install the wheel
pip install dist/spatialx_transform-*.whl
```

## Quick Start

```python
import json
import numpy as np
from spatialx_transform.transforms import Transformation
from spatialx_transform.warp import warp_transform

# 1. Load alignment from JSON
with open("alignment.json") as f:
    aln_data = json.load(f)
align_tf = Transformation.model_validate(aln_data["alignment"]["transformation"])

# 2. Preflight — check output shape and estimated RAM without writing
preflight = warp_transform(
    input_dir="input.zarr",
    output_dir="output_dir",
    tf=align_tf,
    d=(50, 50),
    scale=(0.5, 0.5),
    preflight=True,
)
print(f"Output shape: {preflight.img_shape}, offset: {preflight.offset}")
print(f"Estimated RAM: {preflight.estimated_memory:.2f} GB")

# 3a. Run with default memory management (auto-splits chunks to stay within ~2GB)
result = warp_transform(
    input_dir="input.zarr",
    output_dir="output_dir",
    tf=align_tf,
    d=(50, 50),
    scale=(0.5, 0.5),
    chunk_size=(50000, 50000),
)

# 3b. Or run with explicit output buffer size (controls per-chunk dst_img buffer)
result = warp_transform(
    input_dir="input.zarr",
    output_dir="output_dir",
    tf=align_tf,
    d=(50, 50),
    scale=(0.5, 0.5),
    chunk_size=(50000, 50000),
    output_buffer_size_gb=1.0,
)

# 4. Read output
channel_0 = np.asarray(result.img_zarr_list[0])
```

## `warp_transform()` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_dir` | `str` | *(required)* | Path to input zarr store. Shape `[C, H, W]` (3D) or `[H, W]` (2D). |
| `output_dir` | `str` | *(required)* | Output directory. Writes `img_output_channel{c}.zarr` (3D) or `img_output.zarr` (2D). |
| `tf` | `Transformation` | *(required)* | Transform to apply. Supports `Affine`, `TPS`, `Composed`, `Identity`, `Square`. Load from JSON via `Transformation.model_validate(data)`. |
| `d` | `tuple[int, int]` | `(50, 50)` | Triangle mesh grid step `(dx, dy)`. Smaller = finer mesh = more accurate but slower. |
| `chunk_size` | `tuple[int, int]` | `(10000, 10000)` | Source chunk size `(H, W)`. Controls streaming granularity. Larger = fewer chunks but more memory per chunk. |
| `scale` | `tuple[float, float]` | `(1.0, 1.0)` | Output resolution factor. `<1` downsamples, `>1` upsamples. |
| `preflight` | `bool` | `False` | If `True`, estimates RAM and returns output shape without writing data. |
| `output_buffer_size_gb` | `float \| None` | `None` | Max per-chunk output buffer in GB. If `None`, uses default mode. If set, uses output-buffer mode. |

### Return Types

**`TransformationResult`** (when `preflight=False`):

| Field | Type | Description |
|---|---|---|
| `img_shape` | `list[int]` | Output shape `[C, H_dst, W_dst]` (3D) or `[H_dst, W_dst]` (2D). |
| `img_zarr_list` | `list[zarr.Array]` | List of output zarr arrays, one per channel. Output chunk size is `(512, 512)`. |
| `offset` | `tuple[int, int]` | `(offsetX, offsetY)` of the output bounding box. |

**`PreflightTransformationResult`** (when `preflight=True`):

| Field | Type | Description |
|---|---|---|
| `img_shape` | `list[int]` | Same as `TransformationResult.img_shape`. |
| `offset` | `tuple[int, int]` | Same as `TransformationResult.offset`. |
| `estimated_memory` | `float` | Maximum estimated RAM (GB) across all chunks. |

## Memory Management

The warp transform processes images in chunks to control RAM usage. Two modes control when chunks are split into smaller sub-chunks:

### Output-Buffer Mode (`output_buffer_size_gb=<value>`)

Controls the per-chunk destination image buffer size directly. Chunks whose `dst_img` estimate exceeds `output_buffer_size_gb` are automatically split into `sqrt(K) x sqrt(K)` sub-chunks where `K = dst_img_gb / output_buffer_size_gb`. Run preflight first to learn the total RAM requirement.

### Default Mode (`output_buffer_size_gb=None`)

Uses an internal threshold. Chunks typically stay within ~2GB of RAM (not exact — depends on image dtype, transform complexity, and triangle density). Chunks that exceed the threshold are auto-split the same way.

### Preflight Estimation

Set `preflight=True` to estimate peak RAM without producing output. The returned `estimated_memory` gives the maximum estimated RAM across all chunks, useful for choosing `output_buffer_size_gb`.

### Auto-Splitting

When a chunk's estimate exceeds the threshold, it is recursively split into `sqrt(K) x sqrt(K)` sub-chunks with 1-pixel overlap to ensure seamless triangle meshes at boundaries. Each sub-chunk is processed independently.

## Transformations

```python
from spatialx_transform.transforms import Affine, TPS, Identity, Square, Transformation
from spatialx_transform.params import AffineParams

# Affine: p' = A @ p + b
tf = Affine(params=AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[5.0, 5.0]))

# Identity: p' = p
tf = Identity()

# Square (non-linear, for testing): p' = (x^2, y)
tf = Square()

# Or load any transform from JSON (supports affine, tps, composed, identity, square)
tf = Transformation.model_validate({
    "transformation_type": "affine",
    "params": {"A": [[1.0, 0.0], [0.0, 1.0]], "b": [5.0, 5.0]}
})

# TPS from JSON
tf = Transformation.model_validate({
    "transformation_type": "tps",
    "params": {
        "affine_params": {"A": [[1, 0], [0, 1]], "b": [0, 0]},
        "weights_x": [0.1, -0.05, 0.02],
        "weights_y": [0.03, 0.01, -0.08],
        "control_points": [10, 20, 50, 60, 90, 100]
    }
})

# Composed from JSON (applies transforms in order)
tf = Transformation.model_validate({
    "transformation_type": "composed",
    "params": {
        "transforms": [
            {"transformation_type": "affine", "params": {"A": [[1, 0], [0, 1]], "b": [2, 2]}},
            {"transformation_type": "identity", "params": None}
        ]
    }
})
```

## Recommended Parameters

| Image size (H x W) | `d` (dx, dy) | `chunk_size` | `scale` | `output_buffer_size_gb` |
|---|---|---|---|---|
| 10K x 10K | `(50, 50)` | `(10000, 10000)` | `(1.0, 1.0)` | `None` |
| 50K x 50K | `(50, 50)` | `(25000, 25000)` | `(1.0, 1.0)` | `None` |
| 100K x 30K | `(50, 50)` | `(45000, 15000)` | `(1.0, 1.0)` | `None` |
| 100K x 30K | `(50, 50)` | `(50000, 15000)` | `(0.5, 0.5)` | `1.0` |

## Examples

See `demo.ipynb` for a full walkthrough.
