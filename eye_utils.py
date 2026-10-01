"""
eye_utils.py
------------
Helper functions for eye-landmark based drowsiness detection.

This module contains:
    * MediaPipe Face Landmarker landmark indices that correspond to the
      six points around each eye (matching the classic 6-point EAR
      formulation introduced by Soukupova & Cech, 2016).
    * A function to compute the Eye Aspect Ratio (EAR) from a set
      of 2D landmark points.
    * A small helper to draw the eye landmarks on a video frame for
      visual debugging / demonstration.

The EAR formula used is:

    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

where p1..p6 are the six landmarks around a single eye, ordered as:
    p1 -> left corner
    p2 -> top-left lid point
    p3 -> top-right lid point
    p4 -> right corner
    p5 -> bottom-right lid point
    p6 -> bottom-left lid point

A high EAR indicates an open eye; a low EAR (close to 0) indicates a
closed eye, since the vertical distances collapse while the
horizontal (corner-to-corner) distance stays roughly constant.
"""

from typing import List, Tuple
import numpy as np

# --------------------------------------------------------------------------
# MediaPipe Face Mesh has 468 landmarks. The indices below pick out six
# points around each eye that approximate the dlib 68-point eye contour
# used in the original EAR paper.
# --------------------------------------------------------------------------
LEFT_EYE_IDX: List[int] = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX: List[int] = [33, 160, 158, 133, 153, 144]

# A slightly larger ring of points, used only for drawing a visible
# outline around each eye (not used in the EAR math).
LEFT_EYE_OUTLINE: List[int] = [362, 382, 381, 380, 374, 373, 390, 249,
                                263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_OUTLINE: List[int] = [33, 7, 163, 144, 145, 153, 154, 155,
                                 133, 173, 157, 158, 159, 160, 161, 246]


def _euclidean(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.linalg.norm(np.array(p) - np.array(q)))


def landmarks_to_points(face_landmarks, indices: List[int],
                         image_w: int, image_h: int) -> List[Tuple[float, float]]:
    """
    Convert normalized MediaPipe landmarks (x, y in [0, 1]) into pixel
    coordinates for the requested landmark indices.

    Args:
        face_landmarks: list of normalized landmarks for one face, as returned by
            MediaPipe Tasks FaceLandmarker (`result.face_landmarks[0]`).
        indices: list of landmark indices to extract.
        image_w: width of the source frame, in pixels.
        image_h: height of the source frame, in pixels.

    Returns:
        A list of (x, y) pixel coordinate tuples, in the same order
        as `indices`.
    """
    points = []
    for idx in indices:
        lm = face_landmarks[idx]
        points.append((lm.x * image_w, lm.y * image_h))
    return points


def eye_aspect_ratio(eye_points: List[Tuple[float, float]]) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) given the six (x, y) landmark
    points for a single eye, ordered as described in the module
    docstring (left corner, top-left, top-right, right corner,
    bottom-right, bottom-left).

    Args:
        eye_points: exactly 6 (x, y) pixel coordinate tuples.

    Returns:
        The scalar EAR value. Larger => eye more open.
    """
    if len(eye_points) != 6:
        raise ValueError(f"eye_aspect_ratio expects 6 points, got {len(eye_points)}")

    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = _euclidean(p2, p6)
    vertical_2 = _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def average_ear(face_landmarks, image_w: int, image_h: int) -> Tuple[float, List, List]:
    """
    Compute the average EAR across both eyes for one detected face.

    Returns:
        (avg_ear, left_eye_points, right_eye_points)
    """
    left_pts = landmarks_to_points(face_landmarks, LEFT_EYE_IDX, image_w, image_h)
    right_pts = landmarks_to_points(face_landmarks, RIGHT_EYE_IDX, image_w, image_h)

    left_ear = eye_aspect_ratio(left_pts)
    right_ear = eye_aspect_ratio(right_pts)

    return (left_ear + right_ear) / 2.0, left_pts, right_pts


def draw_eye_outline(frame, face_landmarks, image_w: int, image_h: int):
    """
    Draw a polyline around each eye on `frame` (in-place) for visual
    feedback. Requires OpenCV; imported lazily to keep this module
    importable without a display environment.
    """
    import cv2

    for outline_idx in (LEFT_EYE_OUTLINE, RIGHT_EYE_OUTLINE):
        pts = landmarks_to_points(face_landmarks, outline_idx, image_w, image_h)
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
