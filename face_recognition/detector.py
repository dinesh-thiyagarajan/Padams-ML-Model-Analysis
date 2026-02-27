"""Face detection module using MTCNN."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from face_recognition.utils import crop_face, load_image


@dataclass
class DetectedFace:
    """A single detected face with its metadata."""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    landmarks: Optional[dict] = None
    face_crop: Optional[np.ndarray] = None
    source_image_path: Optional[str] = None


@dataclass
class DetectionResult:
    """Result of face detection on a single image."""

    image_path: str
    faces: List[DetectedFace] = field(default_factory=list)

    @property
    def num_faces(self) -> int:
        return len(self.faces)


class FaceDetector:
    """Face detector using MTCNN.

    MTCNN (Multi-task Cascaded Convolutional Networks) is a robust face
    detector that also predicts facial landmarks. It runs three stages:
    1. P-Net: Proposes candidate face regions
    2. R-Net: Refines the proposals
    3. O-Net: Outputs final bounding boxes + 5-point landmarks
    """

    def __init__(
        self,
        min_face_size: int = 40,
        confidence_threshold: float = 0.9,
        device: str = "cpu",
    ):
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold
        self._detector = None
        self._device = device

    def _load_detector(self):
        """Lazy-load the MTCNN detector."""
        if self._detector is None:
            from mtcnn import MTCNN

            self._detector = MTCNN(
                min_face_size=self.min_face_size,
            )

    def detect_faces(
        self,
        image: np.ndarray,
        image_path: Optional[str] = None,
    ) -> DetectionResult:
        """Detect all faces in an image.

        Args:
            image: RGB image as numpy array (H, W, 3).
            image_path: Optional path to the source image for tracking.

        Returns:
            DetectionResult containing all detected faces.
        """
        self._load_detector()

        result = DetectionResult(image_path=image_path or "")
        detections = self._detector.detect_faces(image)

        for det in detections:
            confidence = det["confidence"]
            if confidence < self.confidence_threshold:
                continue

            # MTCNN returns (x, y, width, height)
            x, y, w, h = det["box"]
            # Convert to (x1, y1, x2, y2) and clamp to image bounds
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(image.shape[1], x + w)
            y2 = min(image.shape[0], y + h)

            face_crop = crop_face(image, (x1, y1, x2, y2), margin=0.2)

            face = DetectedFace(
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
                landmarks=det.get("keypoints"),
                face_crop=face_crop,
                source_image_path=image_path,
            )
            result.faces.append(face)

        return result

    def detect_from_file(self, image_path: str) -> DetectionResult:
        """Detect faces in an image file.

        Args:
            image_path: Path to the image file.

        Returns:
            DetectionResult containing all detected faces.
        """
        image = load_image(image_path)
        return self.detect_faces(image, image_path=image_path)

    def detect_batch(
        self,
        image_paths: List[str],
        show_progress: bool = True,
    ) -> List[DetectionResult]:
        """Detect faces in multiple images.

        Args:
            image_paths: List of paths to image files.
            show_progress: Whether to show a progress bar.

        Returns:
            List of DetectionResult, one per image.
        """
        results = []

        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(image_paths, desc="Detecting faces")
        else:
            iterator = image_paths

        for path in iterator:
            try:
                result = self.detect_from_file(path)
                results.append(result)
            except Exception as e:
                print(f"Warning: Failed to process {path}: {e}")
                results.append(DetectionResult(image_path=path))

        return results
