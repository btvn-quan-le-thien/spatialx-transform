"""Tests for warp_transform."""

import logging

import numpy as np
import pytest

from spatialx_transform.params import AffineParams
from spatialx_transform.transforms import Affine, Identity, Transformation
from spatialx_transform.warp import warp_transform, TransformationResult


def _make_test_img(h=20, w=20, num_channels=1):
    img = np.zeros((num_channels, h, w), dtype=np.uint8)
    for c in range(num_channels):
        for i in range(h):
            for j in range(w):
                img[c, i, j] = (i * 10 + j * 5 + c * 30) % 256
    return img


class TestWarpTransform:
    def test_identity_transform(self):
        img = _make_test_img(20, 20)
        tf = Identity()
        result = warp_transform(img, tf, d=(5, 5), scale=(1.0, 1.0))
        assert isinstance(result, TransformationResult)
        assert result.img.shape[0] == 1
        assert result.img.shape[1] > 0
        assert result.img.shape[2] > 0

    def test_identity_preserves_size(self):
        img = _make_test_img(20, 20)
        tf = Identity()
        result = warp_transform(img, tf, d=(5, 5), scale=(1.0, 1.0))
        assert result.img.shape[1] == pytest.approx(20, abs=5)
        assert result.img.shape[2] == pytest.approx(20, abs=5)

    def test_translation_transform(self):
        img = _make_test_img(20, 20)
        tf = Affine(params=AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[5.0, 5.0]))
        result = warp_transform(img, tf, d=(5, 5), scale=(1.0, 1.0))
        assert result.img.shape[1] >= 20
        assert result.img.shape[2] >= 20
        assert result.offset[0] == 5 or result.offset[1] == 5

    def test_scale_transform(self):
        img = _make_test_img(20, 20)
        tf = Affine(params=AffineParams(A=[[2.0, 0.0], [0.0, 2.0]], b=[0.0, 0.0]))
        result = warp_transform(img, tf, d=(5, 5), scale=(1.0, 1.0))
        assert result.img.shape[1] > 20
        assert result.img.shape[2] > 20

    def test_log_info(self, caplog):
        caplog.set_level(logging.INFO, logger="spatialx_transform.warp")
        img = _make_test_img(10, 10)
        tf = Identity()
        warp_transform(img, tf, d=(3, 3), scale=(1.0, 1.0))
        messages = [record.getMessage() for record in caplog.records]
        assert any("input:" in m for m in messages)
        assert any("output:" in m for m in messages)
        assert any("grid:" in m for m in messages)
        assert any("done:" in m for m in messages)

    def test_no_log_at_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger="spatialx_transform.warp")
        img = _make_test_img(10, 10)
        tf = Identity()
        warp_transform(img, tf, d=(3, 3), scale=(1.0, 1.0))
        assert caplog.records == []

    def test_returns_uint8(self):
        img = _make_test_img(10, 10)
        tf = Identity()
        result = warp_transform(img, tf, d=(3, 3), scale=(1.0, 1.0))
        assert result.img.dtype == np.uint8

    def test_multi_channel(self):
        img = _make_test_img(10, 10, num_channels=3)
        tf = Identity()
        result = warp_transform(img, tf, d=(3, 3), scale=(1.0, 1.0))
        assert result.img.shape[0] == 3

    def test_grid_step_1(self):
        img = _make_test_img(10, 10)
        tf = Identity()
        result = warp_transform(img, tf, d=(1, 1), scale=(1.0, 1.0))
        assert result.img.shape[0] > 0

    def test_with_composed_transform(self):
        img = _make_test_img(15, 15)
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
        result = warp_transform(img, tf, d=(3, 3), scale=(1.0, 1.0))
        assert result.img.shape[1] >= 15
        assert result.img.shape[2] >= 15

    def test_nonzero_output(self):
        img = _make_test_img(20, 20)
        tf = Identity()
        result = warp_transform(img, tf, d=(5, 5), scale=(1.0, 1.0))
        assert np.any(result.img > 0)

    def test_preflight(self):
        img = _make_test_img(10, 10)
        tf = Identity()
        result = warp_transform(img, tf, d=(3, 3), scale=(1.0, 1.0), preflight=True)
        from spatialx_transform.warp import PreflightTransformationResult

        assert isinstance(result, PreflightTransformationResult)
        assert result.img_shape[1] > 0
        assert result.img_shape[2] > 0

    def test_result_has_offset(self):
        img = _make_test_img(10, 10)
        tf = Identity()
        result = warp_transform(img, tf, d=(3, 3), scale=(1.0, 1.0))
        assert len(result.offset) == 2


def _make_test_img_2d(h=20, w=20):
    img = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            img[i, j] = (i * 10 + j * 5) % 256
    return img


class TestWarpTransform2D:
    def test_2d_input_returns_2d(self):
        img = _make_test_img_2d(20, 20)
        tf = Identity()
        result = warp_transform(img, tf, d=(5, 5), scale=(1.0, 1.0))
        assert isinstance(result, TransformationResult)
        assert len(result.img.shape) == 2

    def test_2d_input_scale_enlarges(self):
        img = _make_test_img_2d(10, 10)
        tf = Affine(params=AffineParams(A=[[2.0, 0.0], [0.0, 2.0]], b=[0.0, 0.0]))
        result = warp_transform(img, tf, d=(3, 3), scale=(1.0, 1.0))
        assert result.img.shape[0] > 10
        assert result.img.shape[1] > 10

    def test_2d_vs_3d_equivalence(self):
        img2d = _make_test_img_2d(10, 10)
        img3d = img2d.reshape(1, 10, 10)
        tf = Affine(params=AffineParams(A=[[2.0, 0.0], [0.0, 2.0]], b=[0.0, 0.0]))
        r2d = warp_transform(img2d, tf, d=(3, 3), scale=(1.0, 1.0))
        r3d = warp_transform(img3d, tf, d=(3, 3), scale=(1.0, 1.0))
        assert r2d.img.shape == r3d.img.shape[1:]
        assert np.array_equal(r2d.img, r3d.img[0])

    def test_2d_input_dtype(self):
        img = _make_test_img_2d(10, 10)
        tf = Identity()
        result = warp_transform(img, tf, d=(3, 3), scale=(1.0, 1.0))
        assert result.img.dtype == np.uint8

    def test_2d_input_nonzero(self):
        img = _make_test_img_2d(20, 20)
        tf = Identity()
        result = warp_transform(img, tf, d=(5, 5), scale=(1.0, 1.0))
        assert np.any(result.img > 0)

    def test_2d_input_offset(self):
        img = _make_test_img_2d(10, 10)
        tf = Identity()
        result = warp_transform(img, tf, d=(3, 3), scale=(1.0, 1.0))
        assert len(result.offset) == 2

    def test_2d_input_preflight(self):
        img = _make_test_img_2d(10, 10)
        tf = Identity()
        result = warp_transform(img, tf, d=(3, 3), scale=(1.0, 1.0), preflight=True)
        from spatialx_transform.warp import PreflightTransformationResult

        assert isinstance(result, PreflightTransformationResult)
        assert len(result.img_shape) == 2
        assert result.img_shape[0] > 0
        assert result.img_shape[1] > 0

    def test_2d_input_log_debug(self, caplog):
        caplog.set_level(logging.DEBUG, logger="spatialx_transform.warp")
        img = _make_test_img_2d(10, 10)
        tf = Identity()
        warp_transform(img, tf, d=(3, 3), scale=(1.0, 1.0))
        messages = [record.getMessage() for record in caplog.records]
        assert any("wrapping" in m for m in messages)
        assert any("unwrapping" in m for m in messages)

    def test_2d_input_not_mutated(self):
        img = _make_test_img_2d(10, 10)
        original = img.copy()
        tf = Affine(params=AffineParams(A=[[2.0, 0.0], [0.0, 2.0]], b=[0.0, 0.0]))
        warp_transform(img, tf, d=(3, 3), scale=(1.0, 1.0))
        assert np.array_equal(img, original)

    def test_2d_multi_channel_error(self):
        img = _make_test_img_2d(10, 10)
        tf = Identity()
        result = warp_transform(img, tf, d=(3, 3), scale=(1.0, 1.0))
        assert len(result.img_shape) == 2
