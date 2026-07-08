from dataclasses import dataclass
from typing import overload
from typing import Literal
import logging
import math
import zarr

import cv2 as cv
import numpy as np

from spatialx_transform.point import Point
from spatialx_transform.transforms import Transformation

from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TransformationResult:
    img_shape: list[int]
    img_zarr_list: list[zarr.Array]
    offset: tuple[int, int]


@dataclass
class PreflightTransformationResult:
    img_shape: list[int]
    offset: tuple[int, int]


def _compute_bbox(
    img_shape: tuple[int, int],
    tf: Transformation,
    scale: tuple[float, float],
):
    """
    compute bounding box
    """
    offsetX, offsetY = 1e9, 1e9
    maxX, maxY = -1e9, -1e9

    H_src = img_shape[0]
    W_src = img_shape[1]

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
    return H_dst, W_dst, offsetX, offsetY


def _compute_bbox_for_chunk_dst_img(
    dstTriangle: list[list[Point]],
    scale: tuple[float, float],
    offsetX: int = 0,
    offsetY: int = 0,
    H_dst: int = 0,
    W_dst: int = 0,
):
    """
    compute the bounding box in the chunk output to load the output data to RAM
    """
    minX = 1e9
    minY = 1e9
    maxX = -1e9
    maxY = -1e9
    for i in range(len(dstTriangle)):
        dst_pts = np.array(
            [
                [p.x * scale[1] - offsetY, p.y * scale[0] - offsetX]
                for p in dstTriangle[i]
            ],
            dtype=np.float32,
        )
        x_dst, y_dst, w_dst, h_dst = cv.boundingRect(dst_pts)

        if w_dst == 0 or h_dst == 0:
            continue

        # Clip destination rect to output image bounds
        y0 = max(0, y_dst)
        y1 = min(H_dst, y_dst + h_dst)
        x0 = max(0, x_dst)
        x1 = min(W_dst, x_dst + w_dst)
        if y0 >= y1 or x0 >= x1:
            continue
        minX = min(minX, x0)
        maxX = max(maxX, x1)
        minY = min(minY, y0)
        maxY = max(maxY, y1)

    return minX, minY, maxX, maxY


def _build_chunk_segment(N, c_size):
    chunk = []
    for i in range(0, N, c_size):
        L = i
        R = min(N, i + c_size)
        # overlapping
        if L > 0:
            L = L - 1
        if R + 1 < N:
            R = R + 1
        chunk.append((L, R))
    return chunk


def _warp_transform_chunk_impl(
    img_zarr: zarr.Array,
    tf: Transformation,
    list_img_zarr_output: list[zarr.Array],
    d: tuple[int, int] = (1, 1),
    scale: tuple[float, float] = (1.0, 1.0),
    range_row: tuple[int, int] = (0, 0),
    range_col: tuple[int, int] = (0, 0),
    num_channel: int = 0,
    H_dst: int = 0,
    W_dst: int = 0,
    offsetX: int = 0,
    offsetY: int = 0,
    is2D: bool = False,
):
    """
    process for each chunk [channel, H, W] for chunk
    """

    # load the chunk input
    logger.info("-------> BUILD APRROXIMATE ARRAY AND TRANSPOINT <-------")
    approximated_X = list(range(range_row[0], range_row[1], d[0]))
    approximated_Y = list(range(range_col[0], range_col[1], d[1]))

    if approximated_X[-1] != range_row[1] - 1:
        approximated_X.append(range_row[1] - 1)
    if approximated_Y[-1] != range_col[1] - 1:
        approximated_Y.append(range_col[1] - 1)

    nx, ny = len(approximated_X), len(approximated_Y)
    trans_point = np.zeros((nx, ny), dtype=Point)

    for i in range(nx):
        for j in range(ny):
            x = approximated_X[i]
            y = approximated_Y[j]
            trans_point[i, j] = tf.transform(Point([y, x]))

    logger.info("-------> BUILD TRIANGLE <-------")
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

    logger.info("-------> COMPUTE BBOX FOR CHUNK DST IMG <-------")
    minX, minY, maxX, maxY = _compute_bbox_for_chunk_dst_img(
        dstTriangle=dstTriangle,
        scale=scale,
        offsetX=offsetX,
        offsetY=offsetY,
        H_dst=H_dst,
        W_dst=W_dst,
    )

    if minX > maxX or minY > maxY:
        return

    logger.info("-------> SOLVE FOR CHANNEL <-------")
    for channel in range(num_channel):
        logger.info(f"-------> SOLVE CHANNEL {channel}: <-------")
        logger.info(f"-------> load src img with row = {range_row}, col = {range_col}")
        if not is2D:
            src_img = img_zarr[
                channel, range_row[0] : range_row[1], range_col[0] : range_col[1]
            ]
        else:
            src_img = img_zarr[range_row[0] : range_row[1], range_col[0] : range_col[1]]

        logger.info("-------> load dst img")
        dst_img = list_img_zarr_output[channel][minY:maxY, minX:maxX]

        logger.info("-------> brute sub triangle")
        for i in range(len(srcTriangle)):
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

            src_crop = src_img[
                y_src - range_row[0] : y_src - range_row[0] + h_src,
                x_src - range_col[0] : x_src - range_col[0] + w_src,
            ]

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
            roi = dst_img[y0 - minY : y1 - minY, x0 - minX : x1 - minX]
            mask_roi = mask[
                clip_dy : clip_dy + (y1 - y0), clip_dx : clip_dx + (x1 - x0)
            ]
            warped_roi = warped[
                clip_dy : clip_dy + (y1 - y0), clip_dx : clip_dx + (x1 - x0)
            ]
            idx = mask_roi > 0
            roi[idx] = warped_roi[idx]

        logger.info("-------> store data")
        list_img_zarr_output[channel][minY:maxY, minX:maxX] = dst_img


@overload
def _warp_transform_impl(
    img: zarr.Array,
    output_dir: str,
    tf: Transformation,
    d: tuple[int, int] = (1, 1),
    chunk_size: tuple[int, int] = (1, 1),
    scale: tuple[float, float] = (1.0, 1.0),
    preflight: Literal[False] = ...,
    is2D: bool = False,
) -> TransformationResult: ...


@overload
def _warp_transform_impl(
    img: zarr.Array,
    output_dir: str,
    tf: Transformation,
    d: tuple[int, int] = (1, 1),
    chunk_size: tuple[int, int] = (1, 1),
    scale: tuple[float, float] = (1.0, 1.0),
    preflight: Literal[True] = ...,
    is2D: bool = False,
) -> PreflightTransformationResult: ...


def _warp_transform_impl(
    img: zarr.Array,
    output_dir: str,
    tf: Transformation,
    d: tuple[int, int] = (1, 1),
    chunk_size: tuple[int, int] = (1, 1),
    scale: tuple[float, float] = (1.0, 1.0),
    preflight: bool = False,
    is2D: bool = False,
) -> TransformationResult | PreflightTransformationResult:
    """
    process for image with shape is [C, H, W]
    """
    if preflight:
        logger.info("====================== START PREFLIGHT PHASE =================")
    else:
        logger.info("====================== START WARP_TRANSFORM PHASE ============")

    if is2D:
        num_channel = 1
        H_src = img.shape[0]
        W_src = img.shape[1]
    else:
        num_channel = img.shape[0]
        H_src = img.shape[1]
        W_src = img.shape[2]
    img_dtype = img.dtype
    logger.info(
        f"input: C={num_channel} H={H_src} W={W_src}, d={d}, scale={scale}, is2D = {is2D}"
    )

    H_dst, W_dst, offsetX, offsetY = _compute_bbox(
        img_shape=(H_src, W_src), tf=tf, scale=scale
    )
    logger.info(f"output: H_dst={H_dst} W_dst={W_dst}, offset=({offsetX}, {offsetY})")

    if preflight:
        return PreflightTransformationResult(
            img_shape=[num_channel, H_dst, W_dst] if not is2D else [H_dst, W_dst],
            offset=(offsetX, offsetY),
        )

    logger.info("-------> CREATE ZARR OUTPUT <-------")
    # create zarr output
    list_img_zarr_output = []
    for channel in range(num_channel):
        # spawn zarr output
        zarr_path = (
            str(Path(output_dir) / f"img_output_channel{channel}.zarr")
            if not is2D
            else str(Path(output_dir) / "img_output.zarr")
        )

        img_zarr_ch = zarr.create_array(
            store=zarr_path,
            shape=(H_dst, W_dst),
            chunks=(512, 512),
            dtype=img_dtype,
            fill_value=0,
            overwrite=True,
        )

        list_img_zarr_output.append(img_zarr_ch)

    logger.info("-------> BUILD CHUNK SEGMENT <-------")
    chunk_X = _build_chunk_segment(N=H_src, c_size=chunk_size[0])
    chunk_Y = _build_chunk_segment(N=W_src, c_size=chunk_size[1])

    num_chunk = len(chunk_X) * len(chunk_Y)
    cnt_processed_chunk = 0
    # solve for every chunk
    logger.info("-------> WARP TRANSFORM FOR CHUNK PHASE <-------")
    for i in range(len(chunk_X)):
        for j in range(len(chunk_Y)):
            _warp_transform_chunk_impl(
                img_zarr=img,
                list_img_zarr_output=list_img_zarr_output,
                tf=tf,
                d=d,
                scale=scale,
                range_row=chunk_X[i],
                range_col=chunk_Y[j],
                num_channel=num_channel,
                H_dst=H_dst,
                W_dst=W_dst,
                offsetX=offsetX,
                offsetY=offsetY,
                is2D=is2D,
            )
            cnt_processed_chunk = cnt_processed_chunk + 1
            logger.info(f"Done {cnt_processed_chunk} chunks / {num_chunk} chunks")

    return TransformationResult(
        img_shape=[num_channel, H_dst, W_dst] if not is2D else [H_dst, W_dst],
        img_zarr_list=list_img_zarr_output,
        offset=(offsetX, offsetY),
    )


@overload
def warp_transform(
    input_dir: str,
    output_dir: str,
    tf: Transformation,
    d: tuple[int, int] = (1, 1),
    chunk_size: tuple[int, int] = (1, 1),
    scale: tuple[float, float] = (1.0, 1.0),
    preflight: Literal[False] = ...,
) -> TransformationResult: ...


@overload
def warp_transform(
    input_dir: str,
    output_dir: str,
    tf: Transformation,
    d: tuple[int, int] = (1, 1),
    chunk_size: tuple[int, int] = (1, 1),
    scale: tuple[float, float] = (1.0, 1.0),
    preflight: Literal[True] = ...,
) -> PreflightTransformationResult: ...


def warp_transform(
    input_dir: str,
    output_dir: str,
    tf: Transformation,
    d: tuple[int, int] = (1, 1),
    chunk_size: tuple[int, int] = (1, 1),
    scale: tuple[float, float] = (1.0, 1.0),
    preflight: bool = False,
) -> TransformationResult | PreflightTransformationResult:
    """
    image shape: [channel, H_src, W_src]
    image shape: [H_src, W_src] -> [1, H_src, W_src] -> [1, H_dst, W_dst] -> [H_dst, W_dst]
    """

    img = zarr.open(input_dir, mode="r")

    # logger.info(f"Start warp transform on image with shape {}")
    is2D = len(img.shape) == 2

    if preflight:
        preflight_result = _warp_transform_impl(
            img=img,
            output_dir=output_dir,
            tf=tf,
            d=d,
            scale=scale,
            chunk_size=chunk_size,
            preflight=True,
            is2D=is2D,
        )
        return preflight_result

    result = _warp_transform_impl(
        img=img,
        output_dir=output_dir,
        tf=tf,
        d=d,
        scale=scale,
        chunk_size=chunk_size,
        preflight=False,
        is2D=is2D,
    )

    return result
