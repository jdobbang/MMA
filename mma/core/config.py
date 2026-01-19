"""
Configuration management for MMA pipeline

Central configuration system supporting:
- Dataclass-based configs
- YAML file loading
- Environment variable substitution
- Runtime modification
"""

import os
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class DetectionConfig:
    """Configuration for object detection module"""
    model_path: str = "yolo11x.pt"
    conf_threshold: float = 0.1
    classes: List[int] = field(default_factory=lambda: [0])
    device: str = "cuda"
    batch_size: int = 16
    imgsz: int = 640

    def __post_init__(self):
        """Validate configuration"""
        if self.conf_threshold < 0 or self.conf_threshold > 1:
            raise ValueError("conf_threshold must be between 0 and 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")


@dataclass
class TrackingConfig:
    """Configuration for tracking module"""
    max_age: int = 30
    max_gap: int = 30
    min_score_threshold: float = 0.3
    reid_ema_alpha: float = 0.1
    reid_model_name: str = "osnet_x1_0"
    reid_resize: tuple = field(default_factory=lambda: (256, 128))
    device: str = "cuda"
    max_players: int = 2  # MMA specific

    def __post_init__(self):
        """Validate configuration"""
        if self.max_age < 1:
            raise ValueError("max_age must be >= 1")
        if self.reid_ema_alpha < 0 or self.reid_ema_alpha > 1:
            raise ValueError("reid_ema_alpha must be between 0 and 1")


@dataclass
class PoseConfig:
    """Configuration for pose estimation module"""
    model_path: str = "yolo11x-pose.pt"
    model_type: str = "yolo"  # yolo, rtmpose, vitpose
    keypoint_conf_threshold: float = 0.3
    bbox_padding: float = 0.0
    device: str = "cuda"
    batch_size: int = 16

    def __post_init__(self):
        """Validate configuration"""
        valid_types = ["yolo", "rtmpose", "vitpose"]
        if self.model_type not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}")


@dataclass
class PathConfig:
    """Configuration for data paths with environment variable support"""
    dataset_root: str = "${MMA_DATASET_ROOT:/workspace/MMA/dataset}"
    output_root: str = "${MMA_OUTPUT_ROOT:/workspace/MMA/results}"
    log_root: str = "${MMA_LOG_ROOT:/workspace/MMA/log}"

    def resolve(self) -> None:
        """Resolve environment variables in paths"""
        for field_name in ['dataset_root', 'output_root', 'log_root']:
            value = getattr(self, field_name)
            if value.startswith('${') and ':' in value:
                var_name, default = value[2:-1].split(':', 1)
                resolved_value = os.getenv(var_name, default)
                setattr(self, field_name, resolved_value)

    def __post_init__(self):
        """Resolve paths on initialization"""
        self.resolve()

    def get_dataset_path(self, *args) -> Path:
        """Get path relative to dataset root"""
        return Path(self.dataset_root) / Path(*args)

    def get_output_path(self, *args) -> Path:
        """Get path relative to output root"""
        return Path(self.output_root) / Path(*args)

    def get_log_path(self, *args) -> Path:
        """Get path relative to log root"""
        return Path(self.log_root) / Path(*args)


@dataclass
class MMAConfig:
    """Master configuration class combining all subconfigs"""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MMAConfig":
        """
        Load configuration from YAML file

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            MMAConfig instance

        Raises:
            FileNotFoundError: If YAML file not found
            ValueError: If YAML format is invalid
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {yaml_path}: {e}")

        return cls(
            detection=DetectionConfig(**data.get('detection', {})),
            tracking=TrackingConfig(**data.get('tracking', {})),
            pose=PoseConfig(**data.get('pose', {})),
            paths=PathConfig(**data.get('paths', {}))
        )

    @classmethod
    def from_env(cls, base_config: Optional["MMAConfig"] = None) -> "MMAConfig":
        """
        Create config from environment variables

        Supports environment variables like:
        - MMA_DETECTION_CONF_THRESHOLD
        - MMA_TRACKING_MAX_AGE
        - MMA_POSE_MODEL_PATH

        Args:
            base_config: Base configuration to override (default: new config)

        Returns:
            MMAConfig instance with environment overrides
        """
        if base_config is None:
            config = cls()
        else:
            config = base_config

        # Override detection config
        if 'MMA_DETECTION_CONF_THRESHOLD' in os.environ:
            config.detection.conf_threshold = float(
                os.environ['MMA_DETECTION_CONF_THRESHOLD']
            )
        if 'MMA_DETECTION_DEVICE' in os.environ:
            config.detection.device = os.environ['MMA_DETECTION_DEVICE']

        # Override tracking config
        if 'MMA_TRACKING_MAX_AGE' in os.environ:
            config.tracking.max_age = int(os.environ['MMA_TRACKING_MAX_AGE'])
        if 'MMA_TRACKING_REID_ALPHA' in os.environ:
            config.tracking.reid_ema_alpha = float(
                os.environ['MMA_TRACKING_REID_ALPHA']
            )
        if 'MMA_TRACKING_DEVICE' in os.environ:
            config.tracking.device = os.environ['MMA_TRACKING_DEVICE']

        # Override pose config
        if 'MMA_POSE_DEVICE' in os.environ:
            config.pose.device = os.environ['MMA_POSE_DEVICE']
        if 'MMA_POSE_MODEL_TYPE' in os.environ:
            config.pose.model_type = os.environ['MMA_POSE_MODEL_TYPE']

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)

    def to_yaml(self, yaml_path: str) -> None:
        """
        Save configuration to YAML file

        Args:
            yaml_path: Path to save YAML configuration
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def __str__(self) -> str:
        """String representation of config"""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
