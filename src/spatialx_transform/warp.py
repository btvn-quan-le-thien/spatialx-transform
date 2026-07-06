from dataclasses import dataclass
from typing import overload
from typing import Literal
import math

import cv2 as cv
import numpy as np

from spatialx_transform.point import Point
from spatialx_transform.transforms import Transformation


@dataclass
class TransformationResult:
    img_shape: tuple[int, int, int]
    img: np.ndarray
    offset: tuple[int, int]


@dataclass
class PreflightTransformationResult:
    img_shape: tuple[int, int, int]
    offset: tuple[int, int]


@overload
def warp_transform_main_logic(
    img: np.ndarray,
    tf: Transformation,
    d: tuple[int, int] = (1, 1),
    scale: tuple[float, float] = (1.0, 1.0),
    verbose: bool = False,
    preflight: Literal[False] = ...,
) -> TransformationResult: ...


@overload
def warp_transform_main_logic(
    img: np.ndarray,
    tf: Transformation,
    d: tuple[int, int] = (1, 1),
    scale: tuple[float, float] = (1.0, 1.0),
    verbose: bool = False,
    preflight: Literal[True] = ...,
) -> PreflightTransformationResult: ...


def warp_transform_main_logic(
    img: np.ndarray,
    tf: Transformation,
    d: tuple[int, int] = (1, 1),
    scale: tuple[float, float] = (1.0, 1.0),
    verbose: bool = False,
    preflight: bool = False,
) -> TransformationResult | PreflightTransformationResult:
    """
    process for image with shape is [H, W]
    """

    H_src = img.shape[0]
    W_src = img.shape[1]
    if verbose:
        print(f"[warp] input: {W_src}x{H_src}, dx={d[0]}, dy={d[1]}, scale={scale}")

    offsetX, offsetY = 1e9, 1e9
    maxX, maxY = -1e9, -1e9

    # Boundary — forward scan all border pixels
    if verbose:
        print("[warp] computing forward bbox (border scan)...")
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

    W_dst = np.int32(maxY - offsetY) + 1
    H_dst = np.int32(maxX - offsetX) + 1

    if preflight:
        return PreflightTransformationResult(
            img_shape=(-1, H_dst, W_dst), offset=(offsetX, offsetY)
        )

    img_output = np.zeros((H_dst, W_dst), dtype=np.uint8)
    if verbose:
        print(
            f"[warp] output: {W_dst}x{H_dst}, offset=({math.floor(offsetX)}, {math.floor(offsetY)})"
        )

    approximated_X = list(range(0, H_src, d[0]))
    approximated_Y = list(range(0, W_src, d[1]))

    if approximated_X[-1] != H_src - 1:
        approximated_X.append(H_src - 1)
    if approximated_Y[-1] != W_src - 1:
        approximated_Y.append(W_src - 1)

    nx, ny = len(approximated_X), len(approximated_Y)
    if verbose:
        print(
            f"[warp] grid: {nx}x{ny} = {nx * ny} points, {2 * (nx - 1) * (ny - 1)} triangles"
        )
    trans_point = np.zeros((nx, ny), dtype=Point)

    if verbose:
        print("[warp] forward-transforming grid points...")
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

    if verbose:
        print(f"[warp] warping {len(srcTriangle)} triangles...")
    for i in range(len(srcTriangle)):
        if verbose and i % 200 == 0 and i > 0:
            print(f"  [warp] triangle {i}/{len(srcTriangle)}")
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

        src_crop = img[y_src : y_src + h_src, x_src : x_src + w_src]

        warped = cv.warpAffine(
            src_crop,
            M,
            (w_dst, h_dst),
            flags=cv.INTER_NEAREST | cv.WARP_INVERSE_MAP,
            borderMode=cv.BORDER_REFLECT101,
        )

        mask = np.zeros((h_dst, w_dst), dtype=np.uint8)
        cv.fillConvexPoly(mask, np.int32(dst_local), 255, cv.LINE_AA)

        # Clip destination rect to output image bounds
        y0 = max(0, y_dst)
        y1 = min(H_dst, y_dst + h_dst)
        x0 = max(0, x_dst)
        x1 = min(W_dst, x_dst + w_dst)
        if y0 >= y1 or x0 >= x1:
            continue
        clip_dy = y0 - y_dst
        clip_dx = x0 - x_dst
        roi = img_output[y0:y1, x0:x1]
        mask_roi = mask[clip_dy : clip_dy + (y1 - y0), clip_dx : clip_dx + (x1 - x0)]
        warped_roi = warped[
            clip_dy : clip_dy + (y1 - y0), clip_dx : clip_dx + (x1 - x0)
        ]
        idx = mask_roi > 0
        roi[idx] = warped_roi[idx]

    if verbose:
        print(f"[warp] done: {W_dst}x{H_dst} output")

    return TransformationResult(
        img_shape=(-1, H_dst, W_dst), img=img_output, offset=(offsetX, offsetY)
    )


@overload
def warp_transform(
    img: np.ndarray,
    tf: Transformation,
    d: tuple[int, int] = (1, 1),
    scale: tuple[float, float] = (1.0, 1.0),
    verbose: bool = False,
    preflight: Literal[False] = ...,
) -> TransformationResult: ...


@overload
def warp_transform(
    img: np.ndarray,
    tf: Transformation,
    d: tuple[int, int] = (1, 1),
    scale: tuple[float, float] = (1.0, 1.0),
    verbose: bool = False,
    preflight: Literal[True] = ...,
) -> PreflightTransformationResult: ...


def warp_transform(
    img: np.ndarray,
    tf: Transformation,
    d: tuple[int, int] = (1, 1),
    scale: tuple[float, float] = (1.0, 1.0),
    verbose: bool = False,
    preflight: bool = False,
) -> TransformationResult | PreflightTransformationResult:
    """
    image shape: [channel, H_src, W_src]
    """
    if preflight:
        return warp_transform_main_logic(
            img[0], tf, d, scale, verbose=verbose, preflight=True
        )

    output = []

    num_channel = img.shape[0]

    for i in range(num_channel):
        channel_output = warp_transform_main_logic(
            img[i], tf, d, scale, verbose=verbose, preflight=False
        )
        output_shape = channel_output.img_shape
        output_offset = channel_output.offset
        output.append(channel_output.img)

    return TransformationResult(
        img_shape=(num_channel, output_shape[1], output_shape[2]),
        img=np.array(output),
        offset=output_offset,
    )
