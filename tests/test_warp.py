"""Tests for warp_transform."""

import numpy as np
import pytest

from spatialx_transform.params import AffineParams
from spatialx_transform.transforms import Affine, Identity, Transformation
from spatialx_transform.warp import warp_transform


def _make_test_img(h=20, w=20):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            img[i, j] = [i * 10 % 256, j * 10 % 256, 128]
    return img


class TestWarpTransform:
    def test_identity_transform(self):
        img = _make_test_img(20, 20)
        tf = Identity()
        result = warp_transform(img, tf, dx=5, dy=5, scale=1.0)
        assert result.shape[2] == 3
        assert result.shape[0] > 0
        assert result.shape[1] > 0

    def test_identity_preserves_size(self):
        img = _make_test_img(20, 20)
        tf = Identity()
        result = warp_transform(img, tf, dx=5, dy=5, scale=1.0)
        assert result.shape[0] == pytest.approx(20, abs=5)
        assert result.shape[1] == pytest.approx(20, abs=5)

    def test_translation_transform(self):
        img = _make_test_img(20, 20)
        tf = Affine(params=AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[5.0, 5.0]))
        result = warp_transform(img, tf, dx=5, dy=5, scale=1.0)
        assert result.shape[0] > 20
        assert result.shape[1] > 20

    def test_scale_transform(self):
        img = _make_test_img(20, 20)
        tf = Affine(params=AffineParams(A=[[2.0, 0.0], [0.0, 2.0]], b=[0.0, 0.0]))
        result = warp_transform(img, tf, dx=5, dy=5, scale=1.0)
        assert result.shape[0] > 20
        assert result.shape[1] > 20

    def test_verbose_output(self, capsys):
        img = _make_test_img(10, 10)
        tf = Identity()
        warp_transform(img, tf, dx=3, dy=3, scale=1.0, verbose=True)
        captured = capsys.readouterr()
        assert "[warp] input:" in captured.out
        assert "[warp] output:" in captured.out
        assert "[warp] grid:" in captured.out
        assert "[warp] done:" in captured.out

    def test_no_verbose(self, capsys):
        img = _make_test_img(10, 10)
        tf = Identity()
        warp_transform(img, tf, dx=3, dy=3, scale=1.0, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_returns_uint8(self):
        img = _make_test_img(10, 10)
        tf = Identity()
        result = warp_transform(img, tf, dx=3, dy=3, scale=1.0)
        assert result.dtype == np.uint8

    def test_returns_3_channels(self):
        img = _make_test_img(10, 10)
        tf = Identity()
        result = warp_transform(img, tf, dx=3, dy=3, scale=1.0)
        assert result.shape[2] == 3

    def test_grid_step_1(self):
        img = _make_test_img(10, 10)
        tf = Identity()
        result = warp_transform(img, tf, dx=1, dy=1, scale=1.0)
        assert result.shape[0] > 0

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
        result = warp_transform(img, tf, dx=3, dy=3, scale=1.0)
        assert result.shape[0] > 15
        assert result.shape[1] > 15

    def test_nonzero_output(self):
        img = _make_test_img(20, 20)
        tf = Identity()
        result = warp_transform(img, tf, dx=5, dy=5, scale=1.0)
        assert np.any(result > 0)
