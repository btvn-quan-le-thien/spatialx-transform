from dataclasses import dataclass
from typing import overload
from typing import Literal
import logging
import math

import cv2 as cv
import numpy as np

from spatialx_transform.point import Point
from spatialx_transform.transforms import Transformation

logger = logging.getLogger(__name__)


@dataclass
class TransformationResult:
    img_shape: list[int]
    img: np.ndarray
    offset: tuple[int, int]


@dataclass
class PreflightTransformationResult:
    img_shape: list[int]
    offset: tuple[int, int]


@overload
def _warp_transform_impl(
    img: np.ndarray,
    tf: Transformation,
    d: tuple[int, int] = (1, 1),
    scale: tuple[float, float] = (1.0, 1.0),
    preflight: Literal[False] = ...,
) -> TransformationResult: ...


@overload
def _warp_transform_impl(
    img: np.ndarray,
    tf: Transformation,
    d: tuple[int, int] = (1, 1),
    scale: tuple[float, float] = (1.0, 1.0),
    preflight: Literal[True] = ...,
) -> PreflightTransformationResult: ...


def _warp_transform_impl(
    img: np.ndarray,
    tf: Transformation,
    d: tuple[int, int] = (1, 1),
    scale: tuple[float, float] = (1.0, 1.0),
    preflight: bool = False,
) -> TransformationResult | PreflightTransformationResult:
    """
    process for image with shape is [C, H, W]
    """
    num_channel = img.shape[0]
    H_src = img.shape[1]
    W_src = img.shape[2]
    logger.info(f"input: C={num_channel} H={H_src} W={W_src}, d={d}, scale={scale}")

    offsetX, offsetY = 1e9, 1e9
    maxX, maxY = -1e9, -1e9

    logger.info("computing forward bbox (border scan)...")
    for i in [0, H_src - 1]:
        for j in range(W_src):
            fw_point = tf.transform(Point([j, i]))
            px = fw_point.x * scale[1]
            py = fw_point.y * scale[0]
            offsetX = min(offsetX, py)
            offsetY = min(offsetY, px)
            maxX = max(maxX, py)
            maxY = max(maxY, px)
    for i in range(H_src):
        for j in [0, W_src - 1]:
            fw_point = tf.transform(Point([j, i]))
            px = fw_point.x * scale[1]
            py = fw_point.y * scale[0]
            offsetX = min(offsetX, py)
            offsetY = min(offsetY, px)
            maxX = max(maxX, py)
            maxY = max(maxY, px)

    # rounding offset
    offsetX = math.floor(offsetX)
    offsetY = math.floor(offsetY)

    W_dst = int(maxY - offsetY) + 1
    H_dst = int(maxX - offsetX) + 1
    logger.info(f"output: H_dst={H_dst} W_dst={W_dst}, offset=({offsetX}, {offsetY})")

    if preflight:
        return PreflightTransformationResult(
            img_shape=[num_channel, H_dst, W_dst],
            offset=(offsetX, offsetY),
        )

    img_output = np.zeros((num_channel, H_dst, W_dst), dtype=np.uint8)

    approximated_X = list(range(0, H_src, d[0]))
    approximated_Y = list(range(0, W_src, d[1]))

    if approximated_X[-1] != H_src - 1:
        approximated_X.append(H_src - 1)
    if approximated_Y[-1] != W_src - 1:
        approximated_Y.append(W_src - 1)

    nx, ny = len(approximated_X), len(approximated_Y)
    logger.info(
        f"grid: {nx}x{ny} = {nx * ny} points, {2 * (nx - 1) * (ny - 1)} triangles"
    )
    trans_point = np.zeros((nx, ny), dtype=Point)

    logger.info("forward-transforming grid points...")

    for i in range(nx):
        for j in range(ny):
            x = approximated_X[i]
            y = approximated_Y[j]
            trans_point[i, j] = tf.transform(Point([y, x]))

    srcTriangle = []
    dstTriangle = []

    for i in range(nx - 1):
        for j in range(ny - 1):
            # top - left
            srcTriangle.append(
                [
                    Point([approximated_Y[j], approximated_X[i]]),
                    Point([approximated_Y[j + 1], approximated_X[i]]),
                    Point([approximated_Y[j], approximated_X[i + 1]]),
                ]
            )
            dstTriangle.append(
                [trans_point[i][j], trans_point[i][j + 1], trans_point[i + 1][j]]
            )
            # bottom - right
            srcTriangle.append(
                [
                    Point([approximated_Y[j + 1], approximated_X[i + 1]]),
                    Point([approximated_Y[j + 1], approximated_X[i]]),
                    Point([approximated_Y[j], approximated_X[i + 1]]),
                ]
            )
            dstTriangle.append(
                [
                    trans_point[i + 1][j + 1],
                    trans_point[i][j + 1],
                    trans_point[i + 1][j],
                ]
            )

    logger.info(
        f"warping {len(srcTriangle)} triangles across {num_channel} channel(s)..."
    )
    for channel in range(num_channel):
        src_img = img[channel]
        dst_img = img_output[channel]
        logger.debug(f"processing channel {channel + 1}/{num_channel}")
        for i in range(len(srcTriangle)):
            if i % 20000 == 0 and i > 0:
                logger.debug(f"triangle {i}/{len(srcTriangle)}")
            dst_pts = np.array(
                [
                    [p.x * scale[1] - offsetY, p.y * scale[0] - offsetX]
                    for p in dstTriangle[i]
                ],
                dtype=np.float32,
            )
            src_pts = np.array([[p.x, p.y] for p in srcTriangle[i]], dtype=np.float32)

            x_src, y_src, w_src, h_src = cv.boundingRect(src_pts)
            x_dst, y_dst, w_dst, h_dst = cv.boundingRect(dst_pts)

            if w_src == 0 or h_src == 0 or w_dst == 0 or h_dst == 0:
                continue

            src_local = src_pts - np.array([x_src, y_src], dtype=np.float32)
            dst_local = dst_pts - np.array([x_dst, y_dst], dtype=np.float32)

            M = cv.getAffineTransform(dst_local, src_local)

            src_crop = src_img[y_src : y_src + h_src, x_src : x_src + w_src]

            warped = cv.warpAffine(
                src_crop,
                M,
                (w_dst, h_dst),
                flags=cv.INTER_NEAREST | cv.WARP_INVERSE_MAP,
                borderMode=cv.BORDER_REFLECT101,
            )

            mask = np.zeros((h_dst, w_dst), dtype=np.uint8)
            cv.fillConvexPoly(mask, dst_local.astype(np.int32), 255, cv.LINE_AA)

            # Clip destination rect to output image bounds
            y0 = max(0, y_dst)
            y1 = min(H_dst, y_dst + h_dst)
            x0 = max(0, x_dst)
            x1 = min(W_dst, x_dst + w_dst)
            if y0 >= y1 or x0 >= x1:
                continue
            clip_dy = y0 - y_dst
            clip_dx = x0 - x_dst
            roi = dst_img[y0:y1, x0:x1]
            mask_roi = mask[
                clip_dy : clip_dy + (y1 - y0), clip_dx : clip_dx + (x1 - x0)
            ]
            warped_roi = warped[
                clip_dy : clip_dy + (y1 - y0), clip_dx : clip_dx + (x1 - x0)
            ]
            idx = mask_roi > 0
            roi[idx] = warped_roi[idx]

    logger.info(f"done: {H_dst}x{W_dst} output")

    return TransformationResult(
        img_shape=list(img_output.shape),
        img=img_output,
        offset=(offsetX, offsetY),
    )


@overload
def warp_transform(
    img: np.ndarray,
    tf: Transformation,
    d: tuple[int, int] = (1, 1),
    scale: tuple[float, float] = (1.0, 1.0),
    preflight: Literal[False] = ...,
) -> TransformationResult: ...


@overload
def warp_transform(
    img: np.ndarray,
    tf: Transformation,
    d: tuple[int, int] = (1, 1),
    scale: tuple[float, float] = (1.0, 1.0),
    preflight: Literal[True] = ...,
) -> PreflightTransformationResult: ...


def warp_transform(
    img: np.ndarray,
    tf: Transformation,
    d: tuple[int, int] = (1, 1),
    scale: tuple[float, float] = (1.0, 1.0),
    preflight: bool = False,
) -> TransformationResult | PreflightTransformationResult:
    """
    image shape: [channel, H_src, W_src]
    image shape: [H_src, W_src] -> [1, H_src, W_src] -> [1, H_dst, W_dst] -> [H_dst, W_dst]
    """

    is2D = len(img.shape) == 2
    if is2D:
        logger.debug(f"wrapping 2D input, old shape: {img.shape}")
        img = img[None, ...]
        logger.debug(f"wrapping 2D input, new shape: {img.shape}")

    if preflight:
        preflight_result = _warp_transform_impl(
            img=img, tf=tf, d=d, scale=scale, preflight=True
        )
        if is2D:
            preflight_result.img_shape = preflight_result.img_shape[1:]
        return preflight_result

    result = _warp_transform_impl(img=img, tf=tf, d=d, scale=scale, preflight=False)

    if is2D:
        result.img = result.img.squeeze(axis=0)
        result.img_shape = list(result.img.shape)
        logger.debug(f"unwrapping 2D output, shape: {result.img.shape}")

    return result
