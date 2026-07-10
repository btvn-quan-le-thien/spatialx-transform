"""Tests for warp_transform."""

import logging

import numpy as np
import pytest
import zarr

from spatialx_transform.params import AffineParams
from spatialx_transform.transforms import Affine, Identity, Transformation
from spatialx_transform.warp import (
    TransformationResult,
    PreflightTransformationResult,
    warp_transform,
)


def _make_test_img(h=20, w=20, num_channels=1):
    img = np.zeros((num_channels, h, w), dtype=np.uint8)
    for c in range(num_channels):
        for i in range(h):
            for j in range(w):
                img[c, i, j] = (i * 10 + j * 5 + c * 30) % 256
    return img


def _make_test_img_2d(h=20, w=20):
    img = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            img[i, j] = (i * 10 + j * 5) % 256
    return img


def _write_zarr_input(tmp_path, img, name="input.zarr"):
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
        """Output zarr files should exist on disk after transform."""
        from pathlib import Path

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
        """The input zarr should not be modified by the transform."""
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
        """Transform should work when chunk_size is smaller than the image."""
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
        """Scale factor > 1 should enlarge output relative to input."""
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
        """2D input should produce the same output data as equivalent 3D single-channel input."""
        img2d = _make_test_img_2d(10, 10)
        img3d = img2d.reshape(1, 10, 10)
        input_dir_2d = _write_zarr_input(tmp_path, img2d, name="input2d.zarr")
        input_dir_3d = _write_zarr_input(tmp_path, img3d, name="input3d.zarr")
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
        # 2D shape is [H, W], 3D shape is [C, H, W]; data should match
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
        """2D output should be written to img_output.zarr."""
        from pathlib import Path

        img = _make_test_img_2d(10, 10)
        input_dir = _write_zarr_input(tmp_path, img)
        output_dir = str(tmp_path / "output")
        tf = Identity()
        warp_transform(
            input_dir, output_dir, tf, d=(3, 3), chunk_size=(10, 10), scale=(1.0, 1.0)
        )
        assert (Path(output_dir) / "img_output.zarr").exists()

    def test_2d_log_info(self, tmp_path, caplog):
        """2D path should log is2D=True in the input line."""
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
        """Scale factor > 1 should enlarge 2D output."""
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
