#!/usr/bin/env python3
"""Generate synthetic face-like training data for model validation.

Creates simple synthetic images with distinct face-like patterns for each
"person". This lets us validate the full training pipeline works end-to-end.

For production use, you should train on a real face dataset like:
- LFW (Labeled Faces in the Wild)
- VGGFace2
- MS-Celeb-1M
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_synthetic_face(
    person_id: int,
    variation: int,
    size: int = 112,
    rng: np.random.RandomState = None,
) -> np.ndarray:
    """Generate a synthetic face-like image for a given person.

    Each person gets a unique base pattern (skin tone, eye position, face shape).
    Variations add slight randomness to simulate different photos of the same person.
    """
    if rng is None:
        rng = np.random.RandomState(person_id * 1000 + variation)

    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Unique base properties per person (deterministic from person_id)
    person_rng = np.random.RandomState(person_id * 31 + 7)
    skin_color = person_rng.randint(120, 240, size=3).tolist()
    hair_color = person_rng.randint(20, 100, size=3).tolist()
    eye_color = person_rng.randint(30, 200, size=3).tolist()
    face_width = person_rng.randint(30, 45)
    face_height = person_rng.randint(35, 50)
    eye_offset_y = person_rng.randint(-5, 5)
    nose_len = person_rng.randint(5, 15)
    mouth_width = person_rng.randint(10, 25)

    # Add variation (slightly different for each photo of the same person)
    dx = rng.randint(-3, 4)
    dy = rng.randint(-3, 4)
    brightness = rng.randint(-15, 16)

    cx, cy = size // 2 + dx, size // 2 + dy

    # Background
    bg_color = rng.randint(100, 200, size=3)
    img[:] = bg_color

    # Hair (top of head)
    cv2.ellipse(img, (cx, cy - 10), (face_width + 5, face_height), 0, 180, 360,
                hair_color, -1)

    # Face oval
    skin = [max(0, min(255, c + brightness)) for c in skin_color]
    cv2.ellipse(img, (cx, cy), (face_width, face_height), 0, 0, 360, skin, -1)

    # Eyes
    eye_y = cy - 8 + eye_offset_y
    left_eye_x = cx - face_width // 3
    right_eye_x = cx + face_width // 3
    eye_size = max(3, face_width // 8)

    # Eye whites
    cv2.circle(img, (left_eye_x, eye_y), eye_size + 2, (255, 255, 255), -1)
    cv2.circle(img, (right_eye_x, eye_y), eye_size + 2, (255, 255, 255), -1)

    # Pupils
    pupil_dx = rng.randint(-1, 2)
    cv2.circle(img, (left_eye_x + pupil_dx, eye_y), eye_size, eye_color, -1)
    cv2.circle(img, (right_eye_x + pupil_dx, eye_y), eye_size, eye_color, -1)

    # Eyebrows
    brow_y = eye_y - eye_size - 4
    cv2.line(img, (left_eye_x - eye_size - 2, brow_y),
             (left_eye_x + eye_size + 2, brow_y - 2), hair_color, 2)
    cv2.line(img, (right_eye_x - eye_size - 2, brow_y - 2),
             (right_eye_x + eye_size + 2, brow_y), hair_color, 2)

    # Nose
    nose_top = cy + 2
    cv2.line(img, (cx, nose_top), (cx, nose_top + nose_len),
             [max(0, c - 30) for c in skin], 2)
    cv2.line(img, (cx - 3, nose_top + nose_len), (cx + 3, nose_top + nose_len),
             [max(0, c - 30) for c in skin], 2)

    # Mouth
    mouth_y = cy + nose_len + 10
    mouth_color = [min(255, skin[0] + 40), max(0, skin[1] - 20), max(0, skin[2] - 20)]
    cv2.ellipse(img, (cx, mouth_y), (mouth_width // 2, 4), 0, 0, 180, mouth_color, -1)

    return img


def generate_dataset(
    output_dir: str,
    n_people: int = 20,
    n_images_per_person: int = 15,
    image_size: int = 112,
):
    """Generate a full synthetic dataset.

    Args:
        output_dir: Root directory for the dataset.
        n_people: Number of synthetic people to generate.
        n_images_per_person: Number of images per person.
        image_size: Image size (square).
    """
    os.makedirs(output_dir, exist_ok=True)

    total = 0
    for person_id in range(n_people):
        person_dir = os.path.join(output_dir, f"person_{person_id:03d}")
        os.makedirs(person_dir, exist_ok=True)

        for img_idx in range(n_images_per_person):
            rng = np.random.RandomState(person_id * 10000 + img_idx)
            img = generate_synthetic_face(person_id, img_idx, image_size, rng)

            img_path = os.path.join(person_dir, f"img_{img_idx:03d}.jpg")
            # Convert RGB to BGR for cv2
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            total += 1

    print(f"Generated {total} images for {n_people} people in {output_dir}")
    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic face dataset")
    parser.add_argument("--output_dir", type=str, default="data/synthetic_faces")
    parser.add_argument("--n_people", type=int, default=20)
    parser.add_argument("--n_images", type=int, default=15)
    args = parser.parse_args()

    generate_dataset(args.output_dir, args.n_people, args.n_images)
