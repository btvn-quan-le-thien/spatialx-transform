"""Tests for warp_transform."""

import logging
from pathlib import Path

import numpy as np
import pytest
import zarr

from spatialx_transform.params import AffineParams
from spatialx_transform.transforms import Affine, Identity, Square, Transformation
from spatialx_transform.warp import (
    PreflightTransformationResult,
    TransformationResult,
    WarpContext,
    _build_chunk_segment,
    _build_grid_point,
    _build_triangle_mesh,
    _check_memory_and_split,
    _chunk_aligned_range,
    _compute_bbox,
    _warp_estimated_RAM,
    warp_transform,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_test_img(h=20, w=20, num_channels=1):
    i = np.arange(h, dtype=np.int32)
    j = np.arange(w, dtype=np.int32)
    img = np.zeros((num_channels, h, w), dtype=np.uint8)
    for c in range(num_channels):
        img[c] = ((i[:, None] * 10 + j[None, :] * 5 + c * 30) % 256).astype(np.uint8)
    return img


def _make_test_img_2d(h=20, w=20):
    i = np.arange(h, dtype=np.int32)
    j = np.arange(w, dtype=np.int32)
    return ((i[:, None] * 10 + j[None, :] * 5) % 256).astype(np.uint8)


def _make_test_img_uint16(h=20, w=20, num_channels=1):
    i = np.arange(h, dtype=np.int32)
    j = np.arange(w, dtype=np.int32)
    img = np.zeros((num_channels, h, w), dtype=np.uint16)
    for c in range(num_channels):
        img[c] = ((i[:, None] * 1000 + j[None, :] * 500 + c * 3000) % 65536).astype(
            np.uint16
        )
    return img


def _make_test_img_2d_uint16(h=20, w=20):
    i = np.arange(h, dtype=np.int32)
    j = np.arange(w, dtype=np.int32)
    return ((i[:, None] * 1000 + j[None, :] * 500) % 65536).astype(np.uint16)


def _write_zarr_input(tmp_path, img, name="input.sarr"):
    """Write a numpy array as a zarr store on disk and return the path."""
    zarr_path = str(tmp_path / name)
    z = zarr.create_array(
        store=zarr_path,
        shape=img.shape,
        chunks=(512, 512) if img.ndim == 2 else (1, 512, 512),
        dtype=img.dtype,
        fill_value=0,
        overwrite=True,
    )
    z[:] = img
    return zarr_path


def _read_zarr_output(result):
    """Read all output zarr arrays from a TransformationResult into numpy."""
    return [np.asarray(z[:]) for z in result.img_zarr_list]


def _make_warp_context(
    tmp_path,
    img,
    tf=None,
    d=(5, 5),
    scale=(1.0, 1.0),
    preflight=False,
    memory_limit_gb=2,
    output_buffer_size_gb=None,
    list_img_zarr_output=None,
):
    """Build a WarpContext for direct internal function testing."""
    if tf is None:
        tf = Identity()
    input_dir = _write_zarr_input(tmp_path, img)
    img_zarr = zarr.open(input_dir, mode="r")
    is2D = img_zarr.ndim == 2
    if is2D:
        num_channel = 1
        H_src = img_zarr.shape[0]
        W_src = img_zarr.shape[1]
    else:
        num_channel = img_zarr.shape[0]
        H_src = img_zarr.shape[1]
        W_src = img_zarr.shape[2]
    H_dst, W_dst, offsetX, offsetY = _compute_bbox(
        img_shape=(H_src, W_src), tf=tf, d=d, scale=scale
    )
    if list_img_zarr_output is None:
        list_img_zarr_output = []
    return WarpContext(
        img_zarr=img_zarr,
        tf=tf,
        list_img_zarr_output=list_img_zarr_output,
        d=d,
        scale=scale,
        H_dst=H_dst,
        W_dst=W_dst,
        offsetX=offsetX,
        offsetY=offsetY,
        is2D=is2D,
        num_channel=num_channel,
        preflight=preflight,
        memory_limit_gb=memory_limit_gb,
        output_buffer_size_gb=output_buffer_size_gb,
    )


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------


class TestBuildGridPoint:
    """Tests for _build_grid_point."""

    def test_basic_grid(self):
        result = _build_grid_point(0, 100, 25)
        assert result == [0, 25, 50, 75, 99]

    def test_last_point_already_included(self):
        result = _build_grid_point(0, 100, 1)
        assert result[-1] == 99
        assert len(result) == 100

    def test_single_step(self):
        result = _build_grid_point(0, 10, 20)
        assert result == [0, 9]

    def test_step_equals_range(self):
        result = _build_grid_point(0, 10, 10)
        assert result == [0, 9]


class TestBuildChunkSegment:
    """Tests for _build_chunk_segment."""

    def test_no_overlap_first_chunk(self):
        chunks = _build_chunk_segment(L_range=0, R_range=30, c_size=10)
        assert chunks[0] == (0, 11)

    def test_left_overlap(self):
        chunks = _build_chunk_segment(L_range=0, R_range=30, c_size=10)
        assert chunks[1][0] == 9

    def test_right_overlap(self):
        chunks = _build_chunk_segment(L_range=0, R_range=30, c_size=10)
        assert chunks[0][1] == 11

    def test_no_right_overlap_at_end(self):
        chunks = _build_chunk_segment(L_range=0, R_range=30, c_size=10)
        assert chunks[-1][1] == 30

    def test_exact_chunk_count(self):
        chunks = _build_chunk_segment(L_range=0, R_range=30, c_size=10)
        assert len(chunks) == 3


class TestChunkAlignedRange:
    """Tests for _chunk_aligned_range."""

    def test_already_aligned(self):
        assert _chunk_aligned_range(0, 512, 512) == (0, 512)

    def test_not_aligned(self):
        assert _chunk_aligned_range(100, 600, 512) == (0, 1024)

    def test_single_element(self):
        assert _chunk_aligned_range(5, 6, 512) == (0, 512)

    def test_spanning_multiple_chunks(self):
        assert _chunk_aligned_range(0, 1025, 512) == (0, 1536)


class TestComputeBbox:
    """Tests for _compute_bbox."""

    def test_identity_bbox(self):
        tf = Identity()
        H_dst, W_dst, offsetX, offsetY = _compute_bbox(
            img_shape=(20, 20), tf=tf, d=(5, 5), scale=(1.0, 1.0)
        )
        assert H_dst == pytest.approx(20, abs=2)
        assert W_dst == pytest.approx(20, abs=2)

    def test_scale_bbox(self):
        tf = Affine(params=AffineParams(A=[[2.0, 0.0], [0.0, 2.0]], b=[0.0, 0.0]))
        H_dst, W_dst, offsetX, offsetY = _compute_bbox(
            img_shape=(20, 20), tf=tf, d=(5, 5), scale=(1.0, 1.0)
        )
        assert H_dst > 20
        assert W_dst > 20

    def test_translation_bbox(self):
        tf = Affine(params=AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[5.0, 5.0]))
        H_dst, W_dst, offsetX, offsetY = _compute_bbox(
            img_shape=(20, 20), tf=tf, d=(5, 5), scale=(1.0, 1.0)
        )
        assert 4 <= offsetX <= 6
        assert 4 <= offsetY <= 6

    def test_bbox_returns_4_ints(self):
        tf = Identity()
        result = _compute_bbox(img_shape=(10, 10), tf=tf, d=(3, 3), scale=(1.0, 1.0))
        assert len(result) == 4
        for v in result:
            assert isinstance(v, int)


class TestWarpEstimatedRAM:
    """Tests for _warp_estimated_RAM."""

    def test_returns_two_floats(self, tmp_path):
        img = _make_test_img(20, 20)
        ctx = _make_warp_context(tmp_path, img)
        result = _warp_estimated_RAM(ctx, (0, 20), (0, 20))
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_positive_for_valid_input(self, tmp_path):
        img = _make_test_img(20, 20)
        ctx = _make_warp_context(tmp_path, img, preflight=True)
        total_gb, dst_img_gb = _warp_estimated_RAM(ctx, (0, 20), (0, 20))
        assert total_gb > 0
        assert dst_img_gb > 0

    def test_dst_img_zero_no_outputs_non_preflight(self, tmp_path):
        img = _make_test_img(20, 20)
        ctx = _make_warp_context(tmp_path, img, preflight=False)
        total_gb, dst_img_gb = _warp_estimated_RAM(ctx, (0, 20), (0, 20))
        assert dst_img_gb == 0

    def test_dst_img_nonzero_preflight(self, tmp_path):
        img = _make_test_img(20, 20)
        ctx = _make_warp_context(tmp_path, img, preflight=True)
        _, dst_img_gb = _warp_estimated_RAM(ctx, (0, 20), (0, 20))
        assert dst_img_gb > 0

    def test_dst_img_nonzero_with_outputs(self, tmp_path):
        img = _make_test_img(20, 20)
        output_dir = str(tmp_path / "output")
        zarr.create_array(
            store=output_dir,
            shape=(20, 20),
            chunks=(512, 512),
            dtype=np.uint8,
            fill_value=0,
            overwrite=True,
        )
        out_zarr = zarr.open(output_dir, mode="r+")
        ctx = _make_warp_context(tmp_path, img, list_img_zarr_output=[out_zarr])
        _, dst_img_gb = _warp_estimated_RAM(ctx, (0, 20), (0, 20))
        assert dst_img_gb > 0

    def test_total_gt_dst_img(self, tmp_path):
        img = _make_test_img(20, 20)
        ctx = _make_warp_context(tmp_path, img, preflight=True)
        total_gb, dst_img_gb = _warp_estimated_RAM(ctx, (0, 20), (0, 20))
        assert total_gb >= dst_img_gb


class TestCheckMemoryAndSplit:
    """Tests for _check_memory_and_split."""

    def test_no_split_under_memory_limit(self, tmp_path):
        img = _make_test_img(10, 10)
        ctx = _make_warp_context(tmp_path, img, memory_limit_gb=999)
        estimated_gb, needs_split = _check_memory_and_split(ctx, (0, 10), (0, 10))
        assert needs_split is False
        assert estimated_gb > 0

    def test_split_over_memory_limit(self, tmp_path):
        img = _make_test_img(50, 50)
        ctx = _make_warp_context(
            tmp_path,
            img,
            d=(5, 5),
            memory_limit_gb=0.001,
            preflight=True,
        )
        estimated_gb, needs_split = _check_memory_and_split(ctx, (0, 50), (0, 50))
        assert needs_split is True

    def test_no_split_under_output_buffer(self, tmp_path):
        img = _make_test_img(10, 10)
        ctx = _make_warp_context(tmp_path, img, output_buffer_size_gb=999)
        estimated_gb, needs_split = _check_memory_and_split(ctx, (0, 10), (0, 10))
        assert needs_split is False

    def test_split_over_output_buffer(self, tmp_path):
        img = _make_test_img(600, 600)
        ctx = _make_warp_context(
            tmp_path,
            img,
            d=(50, 50),
            output_buffer_size_gb=0.0005,
            preflight=True,
        )
        estimated_gb, needs_split = _check_memory_and_split(ctx, (0, 600), (0, 600))
        assert needs_split is True


class TestBuildTriangleMesh:
    """Tests for _build_triangle_mesh."""

    def test_triangle_count(self, tmp_path):
        img = _make_test_img(20, 20)
        ctx = _make_warp_context(tmp_path, img, d=(5, 5))
        src_tri, dst_tri = _build_triangle_mesh(ctx, (0, 20), (0, 20))
        expected = 2 * (5 - 1) * (5 - 1)
        assert len(src_tri) == expected
        assert len(dst_tri) == expected

    def test_src_triangle_count(self, tmp_path):
        img = _make_test_img(20, 20)
        ctx = _make_warp_context(tmp_path, img, d=(5, 5))
        src_tri, dst_tri = _build_triangle_mesh(ctx, (0, 20), (0, 20))
        assert len(src_tri) == 2 * (5 - 1) * (5 - 1)

    def test_dst_triangle_count(self, tmp_path):
        img = _make_test_img(20, 20)
        ctx = _make_warp_context(tmp_path, img, d=(5, 5))
        src_tri, dst_tri = _build_triangle_mesh(ctx, (0, 20), (0, 20))
        assert len(src_tri) == len(dst_tri)

    def test_src_points_match_grid(self, tmp_path):
        img = _make_test_img(20, 20)
        ctx = _make_warp_context(tmp_path, img, d=(5, 5))
        src_tri, _ = _build_triangle_mesh(ctx, (0, 20), (0, 20))
        p0, p1, p2 = src_tri[0]
        assert p0.x == 0 and p0.y == 0
        assert p1.x == 5 and p1.y == 0
        assert p2.x == 0 and p2.y == 5


class TestOutputBufferSize:
    """Tests for output_buffer_size_gb feature."""

    def test_output_buffer_size_none_default(self, tmp_path):
        img = _make_test_img(20, 20)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir,
            output_dir,
            tf,
            d=(5, 5),
            chunk_size=(10, 10),
            scale=(1.0, 1.0),
            output_buffer_size_gb=None,
        )
        assert isinstance(result, TransformationResult)
        assert np.any(_read_zarr_output(result)[0] > 0)

    def test_output_buffer_size_triggers_split(self, tmp_path):
        img = _make_test_img(600, 600)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir,
            output_dir,
            tf,
            d=(50, 50),
            chunk_size=(600, 600),
            scale=(1.0, 1.0),
            output_buffer_size_gb=0.0005,
        )
        assert isinstance(result, TransformationResult)
        out = _read_zarr_output(result)
        assert np.any(out[0] > 0)

    def test_output_buffer_size_preflight(self, tmp_path):
        img = _make_test_img(20, 20)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir,
            output_dir,
            tf,
            d=(5, 5),
            chunk_size=(10, 10),
            scale=(1.0, 1.0),
            preflight=True,
            output_buffer_size_gb=0.001,
        )
        assert isinstance(result, PreflightTransformationResult)
        assert result.estimated_memory > 0

    def test_output_buffer_size_no_split_large_buffer(self, tmp_path):
        img = _make_test_img(20, 20)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir,
            output_dir,
            tf,
            d=(5, 5),
            chunk_size=(10, 10),
            scale=(1.0, 1.0),
            output_buffer_size_gb=100,
        )
        assert isinstance(result, TransformationResult)
        out = _read_zarr_output(result)
        assert np.any(out[0] > 0)


class TestMemoryLimitSplitting:
    """Tests for recursive chunk splitting via tiny memory limits."""

    def test_small_memory_limit_triggers_split(self, tmp_path):
        img = _make_test_img(50, 50)
        output_dir = str(tmp_path / "output")
        out_zarr = zarr.create_array(
            store=output_dir,
            shape=(50, 50),
            chunks=(512, 512),
            dtype=np.uint8,
            fill_value=0,
            overwrite=True,
        )
        ctx = _make_warp_context(
            tmp_path,
            img,
            d=(5, 5),
            memory_limit_gb=0.001,
            list_img_zarr_output=[out_zarr],
        )
        from spatialx_transform.warp import _warp_transform_chunk_impl

        _warp_transform_chunk_impl(ctx, (0, 50), (0, 50))
        out = np.asarray(out_zarr[:])
        assert np.any(out > 0)

    def test_split_output_matches_non_split(self, tmp_path):
        img = _make_test_img(600, 600)
        input_dir = _write_zarr_input(tmp_path, img)
        tf = Identity()

        result_normal = warp_transform(
            input_dir,
            str(tmp_path / "output_normal"),
            tf,
            d=(50, 50),
            chunk_size=(600, 600),
            scale=(1.0, 1.0),
        )
        result_split = warp_transform(
            input_dir,
            str(tmp_path / "output_split"),
            tf,
            d=(50, 50),
            chunk_size=(600, 600),
            scale=(1.0, 1.0),
            output_buffer_size_gb=0.0005,
        )
        out_normal = _read_zarr_output(result_normal)
        out_split = _read_zarr_output(result_split)
        assert out_normal[0].shape == out_split[0].shape
        assert np.array_equal(out_normal[0], out_split[0])

    def test_split_log_messages(self, tmp_path, caplog):
        caplog.set_level(logging.INFO, logger="spatialx_transform.warp")
        img = _make_test_img(600, 600)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        warp_transform(
            input_dir,
            output_dir,
            tf,
            d=(50, 50),
            chunk_size=(600, 600),
            scale=(1.0, 1.0),
            output_buffer_size_gb=0.0005,
        )
        messages = [record.getMessage() for record in caplog.records]
        assert any("Splitting chunk" in m for m in messages)


class TestWarpContext:
    """Tests for WarpContext dataclass."""

    def test_construction(self, tmp_path):
        img = _make_test_img(20, 20)
        ctx = _make_warp_context(tmp_path, img)
        assert ctx.img_zarr is not None
        assert ctx.tf is not None
        assert ctx.list_img_zarr_output == []
        assert ctx.d == (5, 5)
        assert ctx.scale == (1.0, 1.0)
        assert isinstance(ctx.H_dst, int)
        assert isinstance(ctx.W_dst, int)
        assert isinstance(ctx.offsetX, int)
        assert isinstance(ctx.offsetY, int)
        assert ctx.is2D is False
        assert ctx.num_channel == 1
        assert ctx.preflight is False
        assert ctx.memory_limit_gb == 2
        assert ctx.output_buffer_size_gb is None

    def test_default_list_img_zarr_output(self, tmp_path):
        img = _make_test_img(10, 10)
        ctx = _make_warp_context(tmp_path, img)
        assert ctx.list_img_zarr_output == []
        assert len(ctx.list_img_zarr_output) == 0


# ---------------------------------------------------------------------------
# Integration tests for 3D input (C, H, W)
# ---------------------------------------------------------------------------


class TestWarpTransform3D:
    """Tests for 3D input (C, H, W)."""

    def test_identity_transform(self, tmp_path):
        img = _make_test_img(20, 20)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir, output_dir, tf, d=(5, 5), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert isinstance(result, TransformationResult)
        assert len(result.img_zarr_list) == 1
        assert result.img_shape[0] == 1
        assert result.img_shape[1] > 0
        assert result.img_shape[2] > 0

    def test_identity_preserves_size(self, tmp_path):
        img = _make_test_img(20, 20)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir, output_dir, tf, d=(5, 5), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert result.img_shape[1] == pytest.approx(20, abs=5)
        assert result.img_shape[2] == pytest.approx(20, abs=5)

    def test_identity_output_correctness(self, tmp_path):
        img = _make_test_img(20, 20)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir, output_dir, tf, d=(5, 5), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        out = np.asarray(result.img_zarr_list[0][:])
        assert out.shape[0] >= 20
        assert out.shape[1] >= 20
        # Identity transform: output should contain original pixel values
        # at the corresponding positions (within bbox offset)
        offsetX, offsetY = result.offset
        for i in range(min(20, out.shape[0] - offsetX)):
            for j in range(min(20, out.shape[1] - offsetY)):
                assert out[i + offsetX, j + offsetY] == img[0, i, j]

    def test_translation_transform(self, tmp_path):
        img = _make_test_img(20, 20)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Affine(params=AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[5.0, 5.0]))
        result = warp_transform(
            input_dir, output_dir, tf, d=(5, 5), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert result.img_shape[1] >= 20
        assert result.img_shape[2] >= 20
        assert result.offset[0] == 5 or result.offset[1] == 5

    def test_scale_transform(self, tmp_path):
        img = _make_test_img(20, 20)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Affine(params=AffineParams(A=[[2.0, 0.0], [0.0, 2.0]], b=[0.0, 0.0]))
        result = warp_transform(
            input_dir, output_dir, tf, d=(5, 5), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert result.img_shape[1] > 20
        assert result.img_shape[2] > 20

    def test_log_info(self, tmp_path, caplog):
        caplog.set_level(logging.INFO, logger="spatialx_transform.warp")
        img = _make_test_img(10, 10)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        messages = [record.getMessage() for record in caplog.records]
        assert any("Input:" in m for m in messages)
        assert any("Output:" in m for m in messages)
        assert any("Warp transform: processing" in m for m in messages)
        assert any("Done" in m and "chunks" in m for m in messages)

    def test_no_log_at_warning(self, tmp_path, caplog):
        caplog.set_level(logging.WARNING, logger="spatialx_transform.warp")
        img = _make_test_img(10, 10)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert caplog.records == []

    def test_returns_uint8(self, tmp_path):
        img = _make_test_img(10, 10)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert result.img_zarr_list[0].dtype == np.uint8

    def test_uint16_dtype(self, tmp_path):
        img = _make_test_img_uint16(10, 10)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert result.img_zarr_list[0].dtype == np.uint16

    def test_multi_channel(self, tmp_path):
        img = _make_test_img(10, 10, num_channels=3)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert result.img_shape[0] == 3
        assert len(result.img_zarr_list) == 3

    def test_multi_channel_data_correctness(self, tmp_path):
        img = _make_test_img(15, 15, num_channels=3)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        out_arrays = _read_zarr_output(result)
        assert len(out_arrays) == 3
        for ch in range(3):
            assert np.any(out_arrays[ch] > 0)
        # Channels should differ
        assert not np.array_equal(out_arrays[0], out_arrays[1])
        assert not np.array_equal(out_arrays[1], out_arrays[2])

    def test_grid_step_1(self, tmp_path):
        img = _make_test_img(10, 10)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir, output_dir, tf, d=(1, 1), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert result.img_shape[0] > 0

    def test_with_composed_transform(self, tmp_path):
        img = _make_test_img(15, 15)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Transformation.model_validate(
            {
                "transformation_type": "composed",
                "params": {
                    "transforms": [
                        {
                            "transformation_type": "affine",
                            "params": {"A": [[1, 0], [0, 1]], "b": [2, 2]},
                        },
                        {"transformation_type": "identity", "params": None},
                    ]
                },
            }
        )
        result = warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert result.img_shape[1] >= 15
        assert result.img_shape[2] >= 15

    def test_nonzero_output(self, tmp_path):
        img = _make_test_img(20, 20)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir, output_dir, tf, d=(5, 5), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        out_arrays = _read_zarr_output(result)
        assert np.any(out_arrays[0] > 0)

    def test_preflight(self, tmp_path):
        img = _make_test_img(10, 10)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir,
            output_dir,
            tf,
            d=(3, 3),
            chunk_size=(10, 10),
            scale=(1.0, 1.0),
            preflight=True,
        )
        assert isinstance(result, PreflightTransformationResult)
        assert len(result.img_shape) == 3
        assert result.img_shape[1] > 0
        assert result.img_shape[2] > 0

    def test_preflight_estimated_memory_positive(self, tmp_path):
        img = _make_test_img(20, 20)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir,
            output_dir,
            tf,
            d=(3, 3),
            chunk_size=(10, 10),
            scale=(1.0, 1.0),
            preflight=True,
        )
        assert isinstance(result, PreflightTransformationResult)
        assert result.estimated_memory > 0

    def test_result_has_offset(self, tmp_path):
        img = _make_test_img(10, 10)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert len(result.offset) == 2

    def test_output_zarr_files_created(self, tmp_path):
        img = _make_test_img(10, 10, num_channels=2)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert (Path(output_dir) / "img_output_channel0.zarr").exists()
        assert (Path(output_dir) / "img_output_channel1.zarr").exists()

    def test_input_not_mutated(self, tmp_path):
        img = _make_test_img(10, 10)
        input_dir = _write_zarr_input(tmp_path, img)
        original = img.copy()
        output_dir = str(tmp_path / "output")
        tf = Affine(params=AffineParams(A=[[2.0, 0.0], [0.0, 2.0]], b=[0.0, 0.0]))
        warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        z_in = zarr.open(input_dir, mode="r")
        assert np.array_equal(np.asarray(z_in[:]), original)

    def test_chunk_size_smaller_than_image(self, tmp_path):
        img = _make_test_img(20, 20)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(5, 5), scale=(1.0, 1.0)
        )
        out_arrays = _read_zarr_output(result)
        assert out_arrays[0].shape[0] > 0
        assert out_arrays[0].shape[1] > 0

    def test_scale_factor_enlarges_output(self, tmp_path):
        img = _make_test_img(10, 10)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result_no_scale = warp_transform(
            input_dir,
            str(tmp_path / "output_noscale"),
            tf,
            d=(3, 3),
            chunk_size=(10, 10),
            scale=(1.0, 1.0),
        )
        result = warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(2.0, 2.0)
        )
        assert result.img_shape[1] > result_no_scale.img_shape[1]
        assert result.img_shape[2] > result_no_scale.img_shape[2]

    def test_scale_down(self, tmp_path):
        img = _make_test_img(20, 20)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir, output_dir, tf, d=(5, 5), chunk_size=(10, 10), scale=(0.5, 0.5)
        )
        assert result.img_shape[1] < 20
        assert result.img_shape[2] < 20

    def test_non_square_image(self, tmp_path):
        img = _make_test_img(30, 15)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir, output_dir, tf, d=(5, 5), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert result.img_shape[1] != result.img_shape[2]

    def test_negative_translation(self, tmp_path):
        img = _make_test_img(20, 20)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Affine(params=AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[-5.0, -5.0]))
        result = warp_transform(
            input_dir, output_dir, tf, d=(5, 5), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert result.offset[0] <= 0 or result.offset[1] <= 0

    def test_square_transform(self, tmp_path):
        img = _make_test_img(15, 15)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Square()
        result = warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert isinstance(result, TransformationResult)
        assert result.img_shape[1] > 0
        assert result.img_shape[2] > 0

    def test_overwrite_existing_output(self, tmp_path):
        img = _make_test_img(10, 10)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        result = warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert isinstance(result, TransformationResult)
        assert np.any(_read_zarr_output(result)[0] > 0)

    def test_large_d(self, tmp_path):
        img = _make_test_img(10, 10)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir, output_dir, tf, d=(20, 20), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert isinstance(result, TransformationResult)
        assert result.img_shape[1] > 0
        assert result.img_shape[2] > 0

    def test_1x1_image(self, tmp_path):
        img = _make_test_img(1, 1)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir, output_dir, tf, d=(1, 1), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert isinstance(result, TransformationResult)
        assert result.img_shape[1] > 0
        assert result.img_shape[2] > 0

    def test_chunk_overlap_correctness(self, tmp_path):
        """Chunk boundaries should produce seamless output (no gaps)."""
        img = _make_test_img(20, 20)
        input_dir = _write_zarr_input(tmp_path, img)
        tf = Identity()

        result_small_chunks = warp_transform(
            input_dir,
            str(tmp_path / "output_small"),
            tf,
            d=(5, 5),
            chunk_size=(5, 5),
            scale=(1.0, 1.0),
        )
        result_large_chunk = warp_transform(
            input_dir,
            str(tmp_path / "output_large"),
            tf,
            d=(5, 5),
            chunk_size=(20, 20),
            scale=(1.0, 1.0),
        )
        out_small = _read_zarr_output(result_small_chunks)
        out_large = _read_zarr_output(result_large_chunk)
        assert out_small[0].shape == out_large[0].shape
        assert np.array_equal(out_small[0], out_large[0])


# ---------------------------------------------------------------------------
# Integration tests for 2D input (H, W)
# ---------------------------------------------------------------------------


class TestWarpTransform2D:
    """Tests for 2D input (H, W)."""

    def test_2d_input_returns_2d_shape(self, tmp_path):
        img = _make_test_img_2d(20, 20)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir, output_dir, tf, d=(5, 5), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert isinstance(result, TransformationResult)
        assert len(result.img_shape) == 2
        assert len(result.img_zarr_list) == 1

    def test_2d_input_scale_enlarges(self, tmp_path):
        img = _make_test_img_2d(10, 10)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Affine(params=AffineParams(A=[[2.0, 0.0], [0.0, 2.0]], b=[0.0, 0.0]))
        result = warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert result.img_shape[0] > 10
        assert result.img_shape[1] > 10

    def test_2d_vs_3d_equivalence(self, tmp_path):
        img2d = _make_test_img_2d(10, 10)
        img3d = img2d.reshape(1, 10, 10)
        input_dir_2d = _write_zarr_input(tmp_path, img2d, name="input2d.sarr")
        input_dir_3d = _write_zarr_input(tmp_path, img3d, name="input3d.sarr")
        output_dir_2d = str(tmp_path / "output2d")
        output_dir_3d = str(tmp_path / "output3d")
        tf = Affine(params=AffineParams(A=[[2.0, 0.0], [0.0, 2.0]], b=[0.0, 0.0]))
        r2d = warp_transform(
            input_dir_2d,
            output_dir_2d,
            tf,
            d=(3, 3),
            chunk_size=(10, 10),
            scale=(1.0, 1.0),
        )
        r3d = warp_transform(
            input_dir_3d,
            output_dir_3d,
            tf,
            d=(3, 3),
            chunk_size=(10, 10),
            scale=(1.0, 1.0),
        )
        assert r2d.img_shape == r3d.img_shape[1:]
        out_2d = np.asarray(r2d.img_zarr_list[0][:])
        out_3d = np.asarray(r3d.img_zarr_list[0][:])
        assert out_2d.shape == out_3d.shape
        assert np.array_equal(out_2d, out_3d)

    def test_2d_input_dtype(self, tmp_path):
        img = _make_test_img_2d(10, 10)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert result.img_zarr_list[0].dtype == np.uint8

    def test_2d_uint16_dtype(self, tmp_path):
        img = _make_test_img_2d_uint16(10, 10)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert result.img_zarr_list[0].dtype == np.uint16

    def test_2d_input_nonzero(self, tmp_path):
        img = _make_test_img_2d(20, 20)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir, output_dir, tf, d=(5, 5), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        out = np.asarray(result.img_zarr_list[0][:])
        assert np.any(out > 0)

    def test_2d_input_offset(self, tmp_path):
        img = _make_test_img_2d(10, 10)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert len(result.offset) == 2

    def test_2d_input_preflight(self, tmp_path):
        img = _make_test_img_2d(10, 10)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir,
            output_dir,
            tf,
            d=(3, 3),
            chunk_size=(10, 10),
            scale=(1.0, 1.0),
            preflight=True,
        )
        assert isinstance(result, PreflightTransformationResult)
        assert len(result.img_shape) == 2
        assert result.img_shape[0] > 0
        assert result.img_shape[1] > 0

    def test_2d_input_not_mutated(self, tmp_path):
        img = _make_test_img_2d(10, 10)
        input_dir = _write_zarr_input(tmp_path, img)
        original = img.copy()
        output_dir = str(tmp_path / "output")
        tf = Affine(params=AffineParams(A=[[2.0, 0.0], [0.0, 2.0]], b=[0.0, 0.0]))
        warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        z_in = zarr.open(input_dir, mode="r")
        assert np.array_equal(np.asarray(z_in[:]), original)

    def test_2d_output_zarr_file_created(self, tmp_path):
        img = _make_test_img_2d(10, 10)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert (Path(output_dir) / "img_output.zarr").exists()

    def test_2d_log_info(self, tmp_path, caplog):
        caplog.set_level(logging.INFO, logger="spatialx_transform.warp")
        img = _make_test_img_2d(10, 10)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        messages = [record.getMessage() for record in caplog.records]
        assert any("Input:" in m and "is2D=True" in m for m in messages)

    def test_2d_scale_factor_enlarges_output(self, tmp_path):
        img = _make_test_img_2d(10, 10)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result_no_scale = warp_transform(
            input_dir,
            str(tmp_path / "output_noscale"),
            tf,
            d=(3, 3),
            chunk_size=(10, 10),
            scale=(1.0, 1.0),
        )
        result = warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(2.0, 2.0)
        )
        assert result.img_shape[0] > result_no_scale.img_shape[0]
        assert result.img_shape[1] > result_no_scale.img_shape[1]

    def test_2d_scale_down(self, tmp_path):
        img = _make_test_img_2d(20, 20)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir, output_dir, tf, d=(5, 5), chunk_size=(10, 10), scale=(0.5, 0.5)
        )
        assert result.img_shape[0] < 20
        assert result.img_shape[1] < 20

    def test_2d_non_square_image(self, tmp_path):
        img = _make_test_img_2d(30, 15)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir, output_dir, tf, d=(5, 5), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert result.img_shape[0] != result.img_shape[1]

    def test_2d_negative_translation(self, tmp_path):
        img = _make_test_img_2d(20, 20)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Affine(params=AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[-5.0, -5.0]))
        result = warp_transform(
            input_dir, output_dir, tf, d=(5, 5), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert result.offset[0] <= 0 or result.offset[1] <= 0

    def test_2d_overwrite_existing_output(self, tmp_path):
        img = _make_test_img_2d(10, 10)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        result = warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert isinstance(result, TransformationResult)
        assert np.any(np.asarray(result.img_zarr_list[0][:]) > 0)

    def test_2d_preflight_estimated_memory_positive(self, tmp_path):
        img = _make_test_img_2d(20, 20)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        result = warp_transform(
            input_dir,
            output_dir,
            tf,
            d=(3, 3),
            chunk_size=(10, 10),
            scale=(1.0, 1.0),
            preflight=True,
        )
        assert isinstance(result, PreflightTransformationResult)
        assert result.estimated_memory > 0
