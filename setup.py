#!/usr/bin/env python
"""
Setup configuration for MMA Pipeline package

Installation:
    pip install -e .

Installation with development dependencies:
    pip install -e ".[dev]"
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="mma-pipeline",
    version="0.1.0",
    description="Modular Multi-task Architecture for MMA Athlete Tracking and Pose Estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MMA Project Team",
    author_email="",
    url="https://github.com/your-org/mma-pipeline",
    license="MIT",
    python_requires=">=3.8",

    packages=find_packages(include=["mma*"]),

    # Include configuration files
    package_data={
        "": ["config/*.yaml"],
    },

    # Core dependencies
    install_requires=[
        # Computer Vision
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "pillow>=8.0.0",

        # Deep Learning
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.0",

        # Object Re-identification
        "torchreid>=1.3.0",

        # Scientific computing
        "scipy>=1.7.0",

        # Filtering
        "filterpy>=1.4.5",

        # Data handling
        "pandas>=1.3.0",
        "pyyaml>=5.4.0",

        # Progress bars
        "tqdm>=4.60.0",

        # Visualization
        "matplotlib>=3.4.0",

        # 3D Graphics (SMPL rendering)
        "trimesh>=3.9.0",
        "pyrender>=0.1.45",
    ],

    # Optional dependencies for development and extended functionality
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "isort>=5.9.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "mmpose": [
            "mmcv>=1.4.0",
            "mmpose>=0.25.0",
            "mmdet>=2.20.0",
            "mmengine>=0.1.0",
        ],
        "all": [
            # This will include all optional dependencies
        ],
    },

    # Entry points for CLI tools
    entry_points={
        "console_scripts": [
            "mma-detect=mma.cli:detect_main",
            "mma-track=mma.cli:track_main",
            "mma-pose=mma.cli:pose_main",
            "mma-pipeline=mma.cli:pipeline_main",
        ],
    },

    # Metadata
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],

    keywords="tracking pose-estimation YOLO SMPL athlete-tracking",

    project_urls={
        "Bug Reports": "https://github.com/your-org/mma-pipeline/issues",
        "Source": "https://github.com/your-org/mma-pipeline",
    },
)
