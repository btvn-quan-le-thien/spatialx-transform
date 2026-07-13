"""Custom exceptions for spatialx_transform."""


class WarpError(Exception):
    """Base exception for all warp-related errors."""


class OpenZarrError(WarpError):
    """Failed to open the input zarr store."""


class CreateOutputError(WarpError):
    """Failed to create output zarr arrays."""


class InvalidImageDimensionError(WarpError):
    """Input image has invalid or unexpected dimensions."""

    def __init__(self, message, shape=None):
        super().__init__(message)
        self.shape = shape


class BuildChunkSegmentError(WarpError):
    """Invalid parameters for chunk segment building."""

    def __init__(self, message, L_range=None, R_range=None, c_size=None):
        super().__init__(message)
        self.L_range = L_range
        self.R_range = R_range
        self.c_size = c_size


class InvalidTransformError(WarpError):
    """Transform produced invalid results (NaN, Inf, or did not converge)."""

    def __init__(self, message, grid_point=None):
        super().__init__(message)
        self.grid_point = grid_point


class ChunkProcessingError(WarpError):
    """A chunk failed during warp processing."""

    def __init__(self, message, chunk_coords=None, estimated_gb=None):
        super().__init__(message)
        self.chunk_coords = chunk_coords
        self.estimated_gb = estimated_gb

    def __str__(self):
        base = super().__str__()
        parts = [base]
        if self.chunk_coords:
            parts.append(f"chunk_coords={self.chunk_coords}")
        if self.estimated_gb is not None:
            parts.append(f"estimated={self.estimated_gb:.3f}GB")
        return " | ".join(parts)


class DegenerateTriangleError(ChunkProcessingError):
    """A triangle was degenerate (zero area or OpenCV cannot warp it)."""

    def __init__(self, message, triangle_index=None, chunk_coords=None):
        super().__init__(message, chunk_coords=chunk_coords)
        self.triangle_index = triangle_index
