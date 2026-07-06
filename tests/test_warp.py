"""Tests for warp_transform."""

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

    def test_verbose_output(self, capsys):
        img = _make_test_img(10, 10)
        tf = Identity()
        warp_transform(img, tf, d=(3, 3), scale=(1.0, 1.0), verbose=True)
        captured = capsys.readouterr()
        assert "[warp] input:" in captured.out
        assert "[warp] output:" in captured.out
        assert "[warp] grid:" in captured.out
        assert "[warp] done:" in captured.out

    def test_no_verbose(self, capsys):
        img = _make_test_img(10, 10)
        tf = Identity()
        warp_transform(img, tf, d=(3, 3), scale=(1.0, 1.0), verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

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
