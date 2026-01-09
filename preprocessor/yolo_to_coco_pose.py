#!/usr/bin/env python3
"""
Convert YOLO pose keypoint format to COCO JSON format.

YOLO Format:
    - One .txt file per image in labels directory
    - Each line contains: class_id x_center y_center width height kpt1_x kpt1_y kpt1_conf ... kptN_x kptN_y kptN_conf
    - All coordinates are normalized (0-1)
    - Confidence values: 0=not visible, 1=occluded, 2=visible

COCO Format:
    - Single JSON file with images, annotations, and categories
    - Bounding box: [x_min, y_min, width, height] in pixel coordinates
    - Keypoints: [x1, y1, v1, x2, y2, v2, ...] in pixel coordinates
    - Visibility: 0=not visible, 1=occluded, 2=visible
"""

import json
import os
import sys
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from PIL import Image


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BBox:
    """Bounding box representation."""
    x_min: float
    y_min: float
    width: float
    height: float

    def to_list(self) -> List[float]:
        """Convert to COCO format [x_min, y_min, width, height]."""
        return [self.x_min, self.y_min, self.width, self.height]

    def area(self) -> float:
        """Calculate bounding box area."""
        return self.width * self.height


@dataclass
class Keypoint:
    """Keypoint representation."""
    x: float
    y: float
    visibility: int

    def to_list(self) -> List[float]:
        """Convert to COCO format [x, y, visibility]."""
        return [self.x, self.y, self.visibility]


class YOLOPoseParser:
    """Parse YOLO pose format lines."""

    NUM_KEYPOINTS = 17
    EXPECTED_VALUES_PER_LINE = 5 + (NUM_KEYPOINTS * 3)  # bbox + keypoints

    @staticmethod
    def parse_line(line: str) -> Tuple[int, BBox, List[Keypoint]]:
        """
        Parse a YOLO pose format line.

        Args:
            line: A line from YOLO format .txt file

        Returns:
            Tuple of (class_id, bbox, keypoints)

        Raises:
            ValueError: If line format is invalid
        """
        values = line.strip().split()

        if len(values) != YOLOPoseParser.EXPECTED_VALUES_PER_LINE:
            raise ValueError(
                f"Expected {YOLOPoseParser.EXPECTED_VALUES_PER_LINE} values, "
                f"got {len(values)}"
            )

        try:
            values = [float(v) for v in values]
        except ValueError as e:
            raise ValueError(f"Could not parse values as floats: {e}")

        class_id = int(values[0])
        x_center = values[1]
        y_center = values[2]
        width = values[3]
        height = values[4]

        # Validate normalized coordinates
        for coord in [x_center, y_center, width, height]:
            if not (0 <= coord <= 1):
                raise ValueError(f"Normalized coordinate out of range: {coord}")

        # Create bounding box (convert from center to top-left)
        bbox = BBox(
            x_min=x_center - width / 2,
            y_min=y_center - height / 2,
            width=width,
            height=height
        )

        # Parse keypoints
        keypoints = []
        kpt_start = 5
        for i in range(YOLOPoseParser.NUM_KEYPOINTS):
            kpt_idx = kpt_start + (i * 3)
            x = values[kpt_idx]
            y = values[kpt_idx + 1]
            visibility = int(values[kpt_idx + 2])

            # Validate coordinates
            if not (0 <= x <= 1) or not (0 <= y <= 1):
                raise ValueError(f"Normalized keypoint coordinate out of range")

            # Validate visibility
            if visibility not in [0, 1, 2]:
                raise ValueError(f"Invalid visibility value: {visibility}")

            keypoints.append(Keypoint(x=x, y=y, visibility=visibility))

        return class_id, bbox, keypoints


class COCOConverter:
    """Convert parsed YOLO format to COCO JSON format."""

    # COCO Pose Keypoints (17 keypoints)
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    # COCO Skeleton connections
    SKELETON = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
        [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
        [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
        [2, 4], [3, 5], [4, 6], [5, 7]
    ]

    @staticmethod
    def create_image_entry(image_id: int, file_name: str, width: int, height: int) -> Dict:
        """Create COCO image entry."""
        return {
            'id': image_id,
            'file_name': file_name,
            'width': width,
            'height': height
        }

    @staticmethod
    def create_annotation_entry(
        annotation_id: int,
        image_id: int,
        bbox: BBox,
        keypoints: List[Keypoint],
        width: int,
        height: int
    ) -> Dict:
        """Create COCO annotation entry."""
        # Convert normalized coordinates to pixel coordinates
        pixel_bbox = BBox(
            x_min=bbox.x_min * width,
            y_min=bbox.y_min * height,
            width=bbox.width * width,
            height=bbox.height * height
        )

        # Convert keypoints to pixel coordinates
        pixel_keypoints = []
        num_visible = 0
        for kpt in keypoints:
            pixel_keypoints.extend([
                kpt.x * width,
                kpt.y * height,
                kpt.visibility
            ])
            if kpt.visibility > 0:
                num_visible += 1

        return {
            'id': annotation_id,
            'image_id': image_id,
            'category_id': 1,  # Person category
            'bbox': pixel_bbox.to_list(),
            'keypoints': pixel_keypoints,
            'num_keypoints': num_visible,
            'area': pixel_bbox.area(),
            'iscrowd': 0
        }

    @staticmethod
    def create_coco_json(
        images: List[Dict],
        annotations: List[Dict]
    ) -> Dict:
        """Create complete COCO format JSON."""
        return {
            'info': {
                'description': 'COCO pose dataset converted from YOLO format',
                'version': '1.0',
                'year': 2024,
                'date_created': '2024'
            },
            'licenses': [
                {
                    'id': 1,
                    'name': 'Unknown',
                    'url': ''
                }
            ],
            'images': images,
            'annotations': annotations,
            'categories': [
                {
                    'id': 1,
                    'name': 'person',
                    'supercategory': 'person',
                    'keypoints': COCOConverter.KEYPOINT_NAMES,
                    'skeleton': COCOConverter.SKELETON
                }
            ]
        }


class DatasetConverter:
    """Main converter orchestrating the entire process."""

    def __init__(self, labels_dir: str, images_dir: str, output_file: str, verbose: bool = False, image_subdir: str = None):
        """
        Initialize the converter.

        Args:
            labels_dir: Path to directory containing YOLO format .txt files
            images_dir: Path to directory containing images
            output_file: Path to output COCO JSON file
            verbose: Enable verbose logging
            image_subdir: Subdirectory to include in file_name (e.g., 'images/train' or 'images/val')
        """
        self.labels_dir = Path(labels_dir)
        self.images_dir = Path(images_dir)
        self.output_file = Path(output_file)
        self.verbose = verbose
        self.image_subdir = image_subdir

        if self.verbose:
            logger.setLevel(logging.DEBUG)

        # Validate directories
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        # Create output directory if needed
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def find_image_file(self, label_filename: str) -> Optional[Path]:
        """
        Find corresponding image file for a label file.

        Args:
            label_filename: Name of label file (with .txt extension)

        Returns:
            Path to image file or None if not found
        """
        base_name = label_filename.replace('.txt', '')
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        for ext in image_extensions:
            image_path = self.images_dir / (base_name + ext)
            if image_path.exists():
                return image_path

        return None

    def get_image_dimensions(self, image_path: Path) -> Tuple[int, int]:
        """
        Get image dimensions.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (width, height)
        """
        try:
            with Image.open(image_path) as img:
                return img.width, img.height
        except Exception as e:
            raise RuntimeError(f"Could not read image dimensions from {image_path}: {e}")

    def convert(self) -> None:
        """Convert entire dataset from YOLO to COCO format."""
        logger.info(f"Starting conversion from {self.labels_dir} to {self.output_file}")

        images = []
        annotations = []
        image_id_counter = 1
        annotation_id_counter = 1
        error_count = 0
        skip_count = 0

        # Get all label files
        label_files = sorted(self.labels_dir.glob('*.txt'))
        total_files = len(label_files)

        if total_files == 0:
            logger.warning(f"No .txt files found in {self.labels_dir}")
            return

        logger.info(f"Found {total_files} label files")

        for idx, label_file in enumerate(label_files, 1):
            logger.debug(f"Processing {idx}/{total_files}: {label_file.name}")

            try:
                # Find corresponding image
                image_path = self.find_image_file(label_file.name)
                if image_path is None:
                    logger.warning(f"No image found for {label_file.name}, skipping")
                    skip_count += 1
                    continue

                # Get image dimensions
                try:
                    width, height = self.get_image_dimensions(image_path)
                except RuntimeError as e:
                    logger.error(str(e))
                    error_count += 1
                    continue

                # Add image to COCO format
                # Include subdirectory in file_name if specified
                if self.image_subdir:
                    file_name = f"{self.image_subdir}/{image_path.name}"
                else:
                    file_name = image_path.name

                images.append(COCOConverter.create_image_entry(
                    image_id=image_id_counter,
                    file_name=file_name,
                    width=width,
                    height=height
                ))

                # Read and parse label file
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                except Exception as e:
                    logger.error(f"Could not read {label_file}: {e}")
                    error_count += 1
                    image_id_counter += 1
                    continue

                # Parse each line (each line = one person instance)
                for line_idx, line in enumerate(lines):
                    if not line.strip():
                        continue

                    try:
                        class_id, bbox, keypoints = YOLOPoseParser.parse_line(line)

                        # Create annotation
                        annotations.append(COCOConverter.create_annotation_entry(
                            annotation_id=annotation_id_counter,
                            image_id=image_id_counter,
                            bbox=bbox,
                            keypoints=keypoints,
                            width=width,
                            height=height
                        ))
                        annotation_id_counter += 1

                    except ValueError as e:
                        logger.error(f"Error parsing line {line_idx + 1} in {label_file}: {e}")
                        error_count += 1

                image_id_counter += 1

            except Exception as e:
                logger.error(f"Error processing {label_file}: {e}")
                error_count += 1
                image_id_counter += 1

        # Create COCO JSON
        coco_data = COCOConverter.create_coco_json(images, annotations)

        # Write output
        try:
            with open(self.output_file, 'w') as f:
                json.dump(coco_data, f, indent=2)
            logger.info(f"Successfully wrote COCO format JSON to {self.output_file}")
        except Exception as e:
            logger.error(f"Could not write output file: {e}")
            sys.exit(1)

        # Summary
        logger.info("=" * 60)
        logger.info("Conversion Summary")
        logger.info("=" * 60)
        logger.info(f"Total images: {len(images)}")
        logger.info(f"Total annotations: {len(annotations)}")
        logger.info(f"Errors encountered: {error_count}")
        logger.info(f"Files skipped: {skip_count}")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Convert YOLO pose keypoint format to COCO JSON format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python yolo_to_coco_pose.py --labels ./data/labels --images ./data/images --output ./coco.json
  python yolo_to_coco_pose.py --labels ./labels --images ./images --output ./coco.json --verbose
        '''
    )

    parser.add_argument(
        '--labels',
        required=True,
        help='Path to directory containing YOLO format .txt label files'
    )
    parser.add_argument(
        '--images',
        required=True,
        help='Path to directory containing images'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to output COCO format JSON file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--image-subdir',
        type=str,
        default=None,
        help='Subdirectory to include in file_name (e.g., "images/train" or "images/val")'
    )

    args = parser.parse_args()

    try:
        converter = DatasetConverter(
            labels_dir=args.labels,
            images_dir=args.images,
            output_file=args.output,
            verbose=args.verbose,
            image_subdir=args.image_subdir
        )
        converter.convert()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
