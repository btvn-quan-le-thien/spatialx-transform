import gc
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
    estimated_memory: float


def _compute_bbox(
    img_shape: tuple[int, int],
    tf: Transformation,
    d: tuple[int, int],
    scale: tuple[float, float],
):
    """
    compute bounding box
    """
    logger.info(f"Computing global destination bounding box via grid scan (d={d}, scale={scale})...")

    offsetX, offsetY = 1e9, 1e9
    maxX, maxY = -1e9, -1e9

    H_src = img_shape[0]
    W_src = img_shape[1]

    approximated_X = list(range(0, H_src, d[0]))
    approximated_Y = list(range(0, W_src, d[1]))
    if approximated_X[-1] != H_src - 1:
        approximated_X.append(H_src - 1)
    if approximated_Y[-1] != W_src - 1:
        approximated_Y.append(W_src - 1)

    for i in approximated_X:
        for j in approximated_Y:
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


def _build_chunk_segment(
    L_range: int = 0,
    R_range: int = 0,
    c_size: int = 0,
) -> list[tuple[int, int]]:
    chunk = []
    for i in range(L_range, R_range, c_size):
        L = i
        R = min(R_range, i + c_size)
        # overlapping
        if L > 0:
            L = L - 1
        if R + 1 <= R_range:
            R = R + 1
        chunk.append((L, R))
    return chunk

def _warp_estimated_RAM(
    img_zarr: zarr.Array,
    tf: Transformation,
    list_img_zarr_output: list[zarr.Array],
    d: tuple[int, int] = (50, 50),
    scale: tuple[float, float] = (1.0, 1.0),
    range_row: tuple[int, int] = (0, 0),
    range_col: tuple[int, int] = (0, 0),
    H_dst: int = 0,
    W_dst: int = 0,
    offsetX: int = 0,
    offsetY: int = 0,
    preflight: bool = False,
) -> tuple[float, float]:
    """
        estimate RAM for chunk
    """
    POINT_SIZE = 850
    LIST_OVERHEAD = 56
    LIST_PTR = 8

    is2D = len(img_zarr.shape) == 2
    src_itemsize = img_zarr.dtype.itemsize
    if is2D:
        src_ch_h, src_ch_w = img_zarr.chunks[0], img_zarr.chunks[1]
    else:
        src_ch_h, src_ch_w = img_zarr.chunks[1], img_zarr.chunks[2]

    approximated_X = list(range(range_row[0], range_row[1], d[0]))
    approximated_Y = list(range(range_col[0], range_col[1], d[1]))
    if approximated_X[-1] != range_row[1] - 1:
        approximated_X.append(range_row[1] - 1)
    if approximated_Y[-1] != range_col[1] - 1:
        approximated_Y.append(range_col[1] - 1)

    nx, ny = len(approximated_X), len(approximated_Y)

    num_tri = 2 * max(0, nx - 1) * max(0, ny - 1)

    if num_tri == 0:
        dst_img_bytes = 0
    else:
        min_px, max_px = 1e9, -1e9
        min_py, max_py = 1e9, -1e9
        for i in range(nx):
            for j in range(ny):
                fw = tf.transform(
                    Point([approximated_Y[j], approximated_X[i]])
                )
                px = fw.x * scale[1] - offsetY
                py = fw.y * scale[0] - offsetX
                min_px = min(min_px, px)
                max_px = max(max_px, px)
                min_py = min(min_py, py)
                max_py = max(max_py, py)

        minX = max(0, math.floor(min_px))
        maxX = min(W_dst, math.ceil(max_px))
        minY = max(0, math.floor(min_py))
        maxY = min(H_dst, math.ceil(max_py))

        if minX > maxX or minY > maxY:
            dst_img_bytes = 0
        elif len(list_img_zarr_output) > 0:
            dst_itemsize = list_img_zarr_output[0].dtype.itemsize
            dst_ch_h = list_img_zarr_output[0].chunks[0]
            dst_ch_w = list_img_zarr_output[0].chunks[1]

            aligned_minY = (minY // dst_ch_h) * dst_ch_h
            aligned_maxY = ((maxY - 1) // dst_ch_h + 1) * dst_ch_h
            aligned_minX = (minX // dst_ch_w) * dst_ch_w
            aligned_maxX = ((maxX - 1) // dst_ch_w + 1) * dst_ch_w

            dst_img_bytes = (
                (aligned_maxY - aligned_minY)
                * (aligned_maxX - aligned_minX)
                * dst_itemsize
            )
        elif preflight:
            dst_itemsize = src_itemsize
            dst_ch_h = 512
            dst_ch_w = 512

            aligned_minY = (minY // dst_ch_h) * dst_ch_h
            aligned_maxY = ((maxY - 1) // dst_ch_h + 1) * dst_ch_h
            aligned_minX = (minX // dst_ch_w) * dst_ch_w
            aligned_maxX = ((maxX - 1) // dst_ch_w + 1) * dst_ch_w

            dst_img_bytes = (
                (aligned_maxY - aligned_minY)
                * (aligned_maxX - aligned_minX)
                * dst_itemsize
            )
        else:
            dst_img_bytes = 0
    trans_point_bytes = nx * ny * POINT_SIZE + nx * ny * LIST_PTR + 200

    src_tri_bytes = num_tri * (LIST_OVERHEAD + 3 * LIST_PTR + 3 * POINT_SIZE)
    dst_tri_bytes = num_tri * (LIST_OVERHEAD + 3 * LIST_PTR)
    outer_list_bytes = 2 * (LIST_OVERHEAD + num_tri * LIST_PTR)
    triangle_bytes = src_tri_bytes + dst_tri_bytes + outer_list_bytes

    r0, r1 = range_row
    c0, c1 = range_col
    aligned_r0 = (r0 // src_ch_h) * src_ch_h
    aligned_r1 = ((r1 - 1) // src_ch_h + 1) * src_ch_h
    aligned_c0 = (c0 // src_ch_w) * src_ch_w
    aligned_c1 = ((c1 - 1) // src_ch_w + 1) * src_ch_w
    src_img_bytes = (aligned_r1 - aligned_r0) * (aligned_c1 - aligned_c0) * src_itemsize

    max_tri_h = int(d[0] * scale[0] * 2) + 2
    max_tri_w = int(d[1] * scale[1] * 2) + 2
    temp_bytes = max_tri_h * max_tri_w * (src_itemsize + 1)

    total_bytes = (
        trans_point_bytes
        + triangle_bytes
        + src_img_bytes
        + dst_img_bytes
        + temp_bytes
    )

    return total_bytes / (1024 ** 3), dst_img_bytes / (1024 ** 3)

def _split_chunk_to_transform(
    img_zarr: zarr.Array,
    tf: Transformation,
    list_img_zarr_output: list[zarr.Array],
    d: tuple[int, int] = (50, 50),
    scale: tuple[float, float] = (1.0, 1.0),
    range_row: tuple[int, int] = (0, 0),
    range_col: tuple[int, int] = (0, 0),
    num_channel: int = 0,
    H_dst: int = 0,
    W_dst: int = 0,
    offsetX: int = 0,
    offsetY: int = 0,
    is2D: bool = False,
    MEMORY_LIMIT_GB: float = 2,
    ESTIMATED_GB: float = 2,
    preflight: bool = False,
    output_buffer_size_gb: float | None = None,
    buffer_size_dst: float = 0,
) -> float:
    """
        estimated > memory_limit. K = estimated / memory_limit
        H_src = range_row[1] - range_col[0]
        W_src = range_row[1] - range_col[0]
        (H_src, W_src) -> (H_src / a, W_src / b) ; a * b = K       
        a = b = sqrt(K)
    """
    logger.info(f"Splitting chunk range_row={range_row}, range_col={range_col}: estimated={ESTIMATED_GB:.3f}GB, limit={MEMORY_LIMIT_GB}GB, dst_img={buffer_size_dst:.3f}GB, output_buffer={output_buffer_size_gb}")
    if output_buffer_size_gb is not None:
        K = buffer_size_dst / output_buffer_size_gb
    else:
        K = ESTIMATED_GB / MEMORY_LIMIT_GB
    a = max(2, int(math.sqrt(K)))
    segment_X = max(2, (range_row[1] - range_row[0]) // a)
    segment_Y = max(2, (range_col[1] - range_col[0]) // a)
    chunk_X = _build_chunk_segment(L_range=range_row[0], R_range=range_row[1], c_size=segment_X)
    chunk_Y = _build_chunk_segment(L_range=range_col[0], R_range=range_col[1], c_size=segment_Y)
    logger.info(f"Sub-chunks: {len(chunk_X)} x {len(chunk_Y)} = {len(chunk_X) * len(chunk_Y)} segments, segment_size=({segment_X}, {segment_Y})")
    max_estimated_GB = 0
    for i in range(len(chunk_X)):
        for j in range(len(chunk_Y)):
            try:
                estimated_gb = _warp_transform_chunk_impl(
                    img_zarr=img_zarr,
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
                    preflight=preflight,
                    output_buffer_size_gb=output_buffer_size_gb,
                )
            except Exception:
                logger.exception(
                    f"Sub-chunk (row={chunk_X[i]}, col={chunk_Y[j]}) failed; skipping"
                )
                continue
            
            max_estimated_GB = max(max_estimated_GB, estimated_gb)
    
    return max_estimated_GB

def _warp_transform_chunk_impl(
    img_zarr: zarr.Array,
    tf: Transformation,
    list_img_zarr_output: list[zarr.Array] = [],
    d: tuple[int, int] = (50, 50),
    scale: tuple[float, float] = (1.0, 1.0),
    range_row: tuple[int, int] = (0, 0),
    range_col: tuple[int, int] = (0, 0),
    num_channel: int = 0,
    H_dst: int = 0,
    W_dst: int = 0,
    offsetX: int = 0,
    offsetY: int = 0,
    is2D: bool = False,
    MEMORY_LIMIT_GB: float = 2,
    preflight: bool = False,
    output_buffer_size_gb: float | None = None,
) -> float:
    """
    process for each chunk [channel, H, W] for chunk
    """
    estimated_gb, dst_img_gb = _warp_estimated_RAM(
        img_zarr=img_zarr,
        tf=tf,
        list_img_zarr_output=list_img_zarr_output,
        d=d,
        scale=scale,
        range_row=range_row,
        range_col=range_col,
        H_dst=H_dst,
        W_dst=W_dst,
        offsetX=offsetX,
        offsetY=offsetY,
        preflight=preflight,
    )
    logger.info(f"Chunk range_row={range_row}, range_col={range_col}: estimated={estimated_gb:.3f}GB, dst_img={dst_img_gb:.3f}GB")
    if (output_buffer_size_gb is not None and dst_img_gb > output_buffer_size_gb) or (output_buffer_size_gb is None and estimated_gb > MEMORY_LIMIT_GB):
        max_estimated_gb = _split_chunk_to_transform(
            img_zarr=img_zarr,
            tf=tf,
            list_img_zarr_output=list_img_zarr_output,
            d=d,
            scale=scale,
            range_row=range_row,
            range_col=range_col,
            num_channel=num_channel,
            H_dst=H_dst,
            W_dst=W_dst,
            offsetX=offsetX,
            offsetY=offsetY,
            is2D=is2D,
            MEMORY_LIMIT_GB=MEMORY_LIMIT_GB,
            ESTIMATED_GB=estimated_gb,
            preflight=preflight,
            output_buffer_size_gb=output_buffer_size_gb,
            buffer_size_dst=dst_img_gb,
        )
        return max_estimated_gb
    if preflight:
        return estimated_gb
    # load the chunk input
    approximated_X = list(range(range_row[0], range_row[1], d[0]))
    approximated_Y = list(range(range_col[0], range_col[1], d[1]))

    if approximated_X[-1] != range_row[1] - 1:
        approximated_X.append(range_row[1] - 1)
    if approximated_Y[-1] != range_col[1] - 1:
        approximated_Y.append(range_col[1] - 1)

    nx, ny = len(approximated_X), len(approximated_Y)
    logger.info(f"Building trans_point matrix: {nx} x {ny} = {nx * ny} grid points")
    trans_point = np.zeros((nx, ny), dtype=Point)

    for i in range(nx):
        for j in range(ny):
            x = approximated_X[i]
            y = approximated_Y[j]
            trans_point[i, j] = tf.transform(Point([y, x]))

    num_tri = 2 * (nx - 1) * (ny - 1)
    logger.info(f"Building triangle mesh: {num_tri} triangles from {nx}x{ny} grid")
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

    minX, minY, maxX, maxY = _compute_bbox_for_chunk_dst_img(
        dstTriangle=dstTriangle,
        scale=scale,
        offsetX=offsetX,
        offsetY=offsetY,
        H_dst=H_dst,
        W_dst=W_dst,
    )
    logger.info(f"Computing dst bbox for chunk: minX={minX}, minY={minY}, maxX={maxX}, maxY={maxY}")

    if minX > maxX or minY > maxY:
        return estimated_gb

    logger.info(f"Processing {num_channel} channels for chunk range_row={range_row}, range_col={range_col}")
    for channel in range(num_channel):
        logger.info(f"Channel {channel}/{num_channel - 1}: loading src_img[row={range_row}, col={range_col}]")
        if not is2D:
            src_img = img_zarr[
                channel, range_row[0] : range_row[1], range_col[0] : range_col[1]
            ]
        else:
            src_img = img_zarr[range_row[0] : range_row[1], range_col[0] : range_col[1]]

        logger.info(f"Channel {channel}: loading dst_img[{minY}:{maxY}, {minX}:{maxX}]")
        dst_img = list_img_zarr_output[channel][minY:maxY, minX:maxX]

        logger.info(f"Channel {channel}: warping {len(srcTriangle)} triangles")
        for i in range(len(srcTriangle)):
            try:
                dst_pts = np.array(
                    [
                        [p.x * scale[1] - offsetY, p.y * scale[0] - offsetX]
                        for p in dstTriangle[i]
                    ],
                    dtype=np.float32,
                )
                src_pts = np.array(
                    [[p.x, p.y] for p in srcTriangle[i]], dtype=np.float32
                )

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
            except cv.error:
                logger.debug(f"Skipping degenerate triangle {i}")

        logger.info(f"Channel {channel}: storing dst_img to zarr and cleaning up")
        # Free triangle loop variables that hold views into src_img/dst_img
        try:
            del src_crop, warped, mask, roi, mask_roi, warped_roi, idx
            del dst_pts, src_pts, src_local, dst_local, M
        except NameError:
            pass

        del src_img
        gc.collect()
        list_img_zarr_output[channel][minY:maxY, minX:maxX] = dst_img
        del dst_img
        gc.collect()
    return estimated_gb

def _warp_preflight(
    img: zarr.Array,
    tf: Transformation,
    d: tuple[int, int] = (50, 50),
    chunk_size: tuple[int, int] = (10000, 10000),
    scale: tuple[float, float] = (1.0, 1.0),
    is2D: bool = False,
    num_channel: int = 0,
    H_dst: int = 0,
    W_dst: int = 0,
    offsetX: int = 0,
    offsetY: int = 0,
    output_buffer_size_gb: float | None = None
) -> PreflightTransformationResult:
    if is2D:
        num_channel = 1
        H_src = img.shape[0]
        W_src = img.shape[1]
    else:
        num_channel = img.shape[0]
        H_src = img.shape[1]
        W_src = img.shape[2]

    chunk_X = _build_chunk_segment(L_range=0, R_range=H_src, c_size=chunk_size[0])
    chunk_Y = _build_chunk_segment(L_range=0, R_range=W_src, c_size=chunk_size[1])
    num_chunk = len(chunk_X) * len(chunk_Y)
    cnt_processed_chunk = 0
    logger.info(f"Preflight: estimating RAM for {num_chunk} chunks (chunk_size={chunk_size})")
    max_estimated_GB = 0
    for i in range(len(chunk_X)):
        for j in range(len(chunk_Y)):
            try:
                estimated_GB = _warp_transform_chunk_impl(
                    img_zarr=img,
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
                    preflight=True,
                    output_buffer_size_gb=output_buffer_size_gb,
                )
            except Exception:
                logger.exception(
                    f"Chunk (row={chunk_X[i]}, col={chunk_Y[j]}) failed; skipping"
                )
                continue
            cnt_processed_chunk = cnt_processed_chunk + 1
            max_estimated_GB = max(max_estimated_GB, estimated_GB)
            logger.info(f"Preflight: done {cnt_processed_chunk}/{num_chunk} chunks, max_estimated={max_estimated_GB:.3f}GB")
    
    return PreflightTransformationResult(
            img_shape=[num_channel, H_dst, W_dst] if not is2D else [H_dst, W_dst],
            offset=(offsetX, offsetY),
            estimated_memory=max_estimated_GB
        )

@overload
def _warp_transform_impl(
    img: zarr.Array,
    output_dir: str,
    tf: Transformation,
    d: tuple[int, int] = (50, 50),
    chunk_size: tuple[int, int] = (10000, 10000),
    scale: tuple[float, float] = (1.0, 1.0),
    preflight: Literal[False] = ...,
    is2D: bool = False,
    output_buffer_size_gb: float | None = None,
) -> TransformationResult: ...


@overload
def _warp_transform_impl(
    img: zarr.Array,
    output_dir: str,
    tf: Transformation,
    d: tuple[int, int] = (50, 50),
    chunk_size: tuple[int, int] = (10000, 10000),
    scale: tuple[float, float] = (1.0, 1.0),
    preflight: Literal[True] = ...,
    is2D: bool = False,
    output_buffer_size_gb: float | None = None,
) -> PreflightTransformationResult: ...


def _warp_transform_impl(
    img: zarr.Array,
    output_dir: str,
    tf: Transformation,
    d: tuple[int, int] = (50, 50),
    chunk_size: tuple[int, int] = (10000, 10000),
    scale: tuple[float, float] = (1.0, 1.0),
    preflight: bool = False,
    is2D: bool = False,
    output_buffer_size_gb: float | None = None
) -> TransformationResult | PreflightTransformationResult:
    """
    process for image with shape is [C, H, W]
    """
    if preflight:
        logger.info("========== START PREFLIGHT PHASE ==========")
    else:
        logger.info("========== START WARP TRANSFORM PHASE ==========")

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
        f"Input: channels={num_channel}, H_src={H_src}, W_src={W_src}, d={d}, scale={scale}, is2D={is2D}"
    )

    H_dst, W_dst, offsetX, offsetY = _compute_bbox(
        img_shape=(H_src, W_src), tf=tf, d=d, scale=scale
    )
    logger.info(f"Output: H_dst={H_dst}, W_dst={W_dst}, offset=({offsetX}, {offsetY})")

    if preflight:
        return _warp_preflight(
            img=img,
            tf=tf,
            d=d,
            chunk_size=chunk_size,
            scale=scale,
            is2D=is2D,
            num_channel=num_channel,
            H_dst=H_dst,
            W_dst=W_dst,
            offsetX=offsetX,
            offsetY=offsetY,
            output_buffer_size_gb=output_buffer_size_gb,
        )

    logger.info(f"Creating {num_channel} output zarr arrays (chunks=(512,512), dtype={img_dtype})")
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

    logger.info(f"Building chunk segments: chunk_size={chunk_size}")
    chunk_X = _build_chunk_segment(L_range=0, R_range=H_src, c_size=chunk_size[0])
    chunk_Y = _build_chunk_segment(L_range=0, R_range=W_src, c_size=chunk_size[1])
    num_chunk = len(chunk_X) * len(chunk_Y)
    cnt_processed_chunk = 0
    # solve for every chunk
    logger.info(f"Warp transform: processing {num_chunk} chunks")
    for i in range(len(chunk_X)):
        for j in range(len(chunk_Y)):
            try:
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
                    preflight=False,
                    output_buffer_size_gb=output_buffer_size_gb,
                )
            except Exception:
                logger.exception(
                    f"Chunk (row={chunk_X[i]}, col={chunk_Y[j]}) failed; skipping"
                )
                continue
            cnt_processed_chunk = cnt_processed_chunk + 1
            logger.info(f"Done {cnt_processed_chunk}/{num_chunk} chunks (row={chunk_X[i]}, col={chunk_Y[j]})")

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
    d: tuple[int, int] = (50, 50),
    chunk_size: tuple[int, int] = (10000, 10000),
    scale: tuple[float, float] = (1.0, 1.0),
    preflight: Literal[False] = ...,
    output_buffer_size_gb: float | None = None
) -> TransformationResult: ...


@overload
def warp_transform(
    input_dir: str,
    output_dir: str,
    tf: Transformation,
    d: tuple[int, int] = (50, 50),
    chunk_size: tuple[int, int] = (10000, 10000),
    scale: tuple[float, float] = (1.0, 1.0),
    preflight: Literal[True] = ...,
    output_buffer_size_gb: float | None = None
) -> PreflightTransformationResult: ...


def warp_transform(
    input_dir: str,
    output_dir: str,
    tf: Transformation,
    d: tuple[int, int] = (50, 50),
    chunk_size: tuple[int, int] = (10000, 10000),
    scale: tuple[float, float] = (1.0, 1.0),
    preflight: bool = False,
    output_buffer_size_gb: float | None = None
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
            output_buffer_size_gb=output_buffer_size_gb,
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
        output_buffer_size_gb=output_buffer_size_gb
    )

    return result
