"""
Custom exceptions for MMA pipeline

Provides specific exception types for:
- Data loading errors
- Model loading errors
- Configuration errors
- Validation errors
"""


class MMAException(Exception):
    """
    Base exception class for all MMA pipeline exceptions

    All custom exceptions should inherit from this class for easy
    exception catching and handling at the application level.
    """
    pass


class ImageLoadError(MMAException):
    """
    Raised when an image fails to load

    Reasons:
    - File does not exist
    - File format is corrupted
    - Insufficient permissions
    - Invalid file path

    Example:
        >>> from mma.core.exceptions import ImageLoadError
        >>> from mma.io import ImageLoader
        >>> try:
        ...     img = ImageLoader.load("nonexistent.jpg")
        ... except ImageLoadError as e:
        ...     print(f"Failed to load image: {e}")
    """
    pass


class DataLoadError(MMAException):
    """
    Raised when data files fail to load

    Applicable to:
    - NPY files (numpy arrays)
    - CSV files
    - SMPL data
    - Poses2D data
    - Bounding box data

    Example:
        >>> from mma.core.exceptions import DataLoadError
        >>> from mma.io import NPYLoader
        >>> try:
        ...     data = NPYLoader.load_smpl("path/to/data.npy", "00001")
        ... except DataLoadError as e:
        ...     print(f"Failed to load data: {e}")
    """
    pass


class ModelLoadError(MMAException):
    """
    Raised when a deep learning model fails to load

    Applicable to:
    - YOLO detection models
    - Pose estimation models
    - Re-ID models
    - SMPL models

    Example:
        >>> from mma.core.exceptions import ModelLoadError
        >>> from mma.detection import YOLODetector
        >>> try:
        ...     detector = YOLODetector("invalid_model.pt")
        ... except ModelLoadError as e:
        ...     print(f"Failed to load model: {e}")
    """
    pass


class ConfigError(MMAException):
    """
    Raised when configuration is invalid or missing

    Reasons:
    - Required configuration key is missing
    - Configuration value is out of valid range
    - Invalid configuration file format
    - Environment variable is not set

    Example:
        >>> from mma.core.exceptions import ConfigError
        >>> from mma.core.config import MMAConfig
        >>> try:
        ...     config = MMAConfig.from_yaml("nonexistent.yaml")
        ... except ConfigError as e:
        ...     print(f"Configuration error: {e}")
    """
    pass


class ValidationError(MMAException):
    """
    Raised when input data validation fails

    Applicable to:
    - Bounding box validation (coordinates, size)
    - Keypoint validation (confidence, visibility)
    - Track validation (ID, frame sequence)
    - Image validation (shape, dtype)

    Example:
        >>> from mma.core.exceptions import ValidationError
        >>> # Validation can raise this error
    """
    pass


class PathError(MMAException):
    """
    Raised when path-related operations fail

    Reasons:
    - Path does not exist
    - Insufficient permissions
    - Invalid path format
    - Directory creation failed

    Example:
        >>> from mma.core.exceptions import PathError
        >>> # Path operations can raise this error
    """
    pass


class InferenceError(MMAException):
    """
    Raised when model inference fails

    Reasons:
    - GPU/CUDA error
    - Input shape mismatch
    - Model not initialized
    - GPU out of memory

    Example:
        >>> from mma.core.exceptions import InferenceError
        >>> # Model inference can raise this error
    """
    pass


class TrackingError(MMAException):
    """
    Raised when tracking algorithm fails

    Reasons:
    - No detections found
    - Kalman filter divergence
    - Re-ID feature extraction failed
    - Invalid track state

    Example:
        >>> from mma.core.exceptions import TrackingError
        >>> # Tracking operations can raise this error
    """
    pass


def handle_mma_exception(e: MMAException, verbose: bool = True) -> str:
    """
    Handle MMA exceptions with formatted error message

    Args:
        e: The MMAException instance
        verbose: If True, print error message to console

    Returns:
        Formatted error message string
    """
    error_type = type(e).__name__
    error_msg = str(e)
    formatted_msg = f"[{error_type}] {error_msg}"

    if verbose:
        print(formatted_msg)

    return formatted_msg
