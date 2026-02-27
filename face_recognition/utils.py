"""Utility functions for face recognition pipeline."""

import os
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk and return as RGB numpy array."""
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def save_image(image: np.ndarray, output_path: str) -> None:
    """Save a numpy array image to disk."""
    img = Image.fromarray(image)
    img.save(output_path)


def get_image_files(directory: str) -> List[str]:
    """Recursively find all image files in a directory."""
    image_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                image_files.append(os.path.join(root, f))
    return sorted(image_files)


def crop_face(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: float = 0.2,
) -> np.ndarray:
    """Crop a face region from an image with optional margin.

    Args:
        image: RGB image as numpy array (H, W, 3).
        bbox: Bounding box as (x1, y1, x2, y2).
        margin: Fractional margin to add around the face.

    Returns:
        Cropped face region as numpy array.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    face_w = x2 - x1
    face_h = y2 - y1

    # Add margin
    margin_x = int(face_w * margin)
    margin_y = int(face_h * margin)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    return image[y1:y2, x1:x2]


def preprocess_face(
    face_image: np.ndarray,
    target_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """Preprocess a face crop for the embedding model.

    Args:
        face_image: RGB face crop as numpy array.
        target_size: Target size for the model input.

    Returns:
        Preprocessed face as float32 array normalized to [-1, 1].
    """
    face = cv2.resize(face_image, target_size)
    face = face.astype(np.float32)
    # Normalize to [-1, 1]
    face = (face - 127.5) / 128.0
    return face


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = a.flatten()
    b = b.flatten()
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute L2 (Euclidean) distance between two vectors."""
    return float(np.linalg.norm(a.flatten() - b.flatten()))
