"""
Visualization module - Rendering detection, tracking, and pose results

Provides:
- Bounding box drawing
- Skeleton visualization
- Color management
- Multi-object rendering
"""

from .drawer import (
    generate_track_color,
    draw_bbox,
    draw_skeleton,
    draw_keypoints,
    draw_multiple_bboxes,
    draw_multiple_skeletons,
    add_text_label,
    overlay_alpha,
)

__all__ = [
    # Color
    "generate_track_color",
    # Drawing
    "draw_bbox",
    "draw_skeleton",
    "draw_keypoints",
    "draw_multiple_bboxes",
    "draw_multiple_skeletons",
    "add_text_label",
    "overlay_alpha",
]
