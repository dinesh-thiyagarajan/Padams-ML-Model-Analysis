"""End-to-end face grouping pipeline.

Ties together detection, embedding, and clustering into a single easy-to-use
interface for grouping photos by person.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from face_recognition.clusterer import FaceClusterer, FaceGroup
from face_recognition.detector import DetectedFace, DetectionResult, FaceDetector
from face_recognition.embedder import FaceEmbedder
from face_recognition.utils import get_image_files, load_image, save_image


@dataclass
class FaceRecord:
    """Complete record for a single detected face."""

    face_index: int
    image_path: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    embedding: Optional[np.ndarray] = None
    group_id: Optional[int] = None


@dataclass
class GroupingResult:
    """Complete result of the face grouping pipeline."""

    face_records: List[FaceRecord] = field(default_factory=list)
    groups: List[FaceGroup] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None

    @property
    def num_faces(self) -> int:
        return len(self.face_records)

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    def get_group_images(self, group_id: int) -> List[str]:
        """Get all unique image paths for a group."""
        for group in self.groups:
            if group.group_id == group_id:
                return group.unique_images
        return []

    def get_images_with_person(self, group_id: int) -> List[str]:
        """Get all images containing a specific person (by group ID)."""
        return self.get_group_images(group_id)

    def to_json(self) -> dict:
        """Convert results to JSON-serializable dict."""
        return {
            "num_faces": self.num_faces,
            "num_groups": self.num_groups,
            "groups": [
                {
                    "group_id": g.group_id,
                    "num_faces": g.num_faces,
                    "images": g.unique_images,
                    "representative_face_index": g.representative_index,
                }
                for g in self.groups
            ],
            "faces": [
                {
                    "index": r.face_index,
                    "image": r.image_path,
                    "bbox": list(r.bbox),
                    "confidence": r.confidence,
                    "group_id": r.group_id,
                }
                for r in self.face_records
            ],
        }

    def save_json(self, output_path: str) -> None:
        """Save results to a JSON file."""
        with open(output_path, "w") as f:
            json.dump(self.to_json(), f, indent=2)


class FaceGroupingPipeline:
    """End-to-end pipeline for grouping photos by the faces in them.

    Usage:
        pipeline = FaceGroupingPipeline()

        # Group all faces in a directory
        result = pipeline.group_faces_in_directory("path/to/photos/")

        # Find all photos of a specific person
        my_photos = pipeline.find_person("path/to/my_face.jpg", "path/to/all_photos/")
    """

    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        embedder: Optional[FaceEmbedder] = None,
        clusterer: Optional[FaceClusterer] = None,
        model_path: Optional[str] = None,
    ):
        self.detector = detector or FaceDetector()
        self.embedder = embedder or FaceEmbedder(model_path=model_path)
        self.clusterer = clusterer or FaceClusterer()

    def group_faces_in_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        show_progress: bool = True,
    ) -> GroupingResult:
        """Process all images in a directory and group faces by person.

        Args:
            input_dir: Directory containing images.
            output_dir: Optional directory to save grouped results.
            show_progress: Whether to show progress bars.

        Returns:
            GroupingResult with all faces grouped by person.
        """
        # 1. Find all images
        image_paths = get_image_files(input_dir)
        if not image_paths:
            print(f"No images found in {input_dir}")
            return GroupingResult()

        print(f"Found {len(image_paths)} images")

        # 2. Detect faces
        print("Step 1/3: Detecting faces...")
        detection_results = self.detector.detect_batch(
            image_paths, show_progress=show_progress
        )

        # 3. Collect all faces and extract embeddings
        print("Step 2/3: Extracting face embeddings...")
        all_faces: List[DetectedFace] = []
        for result in detection_results:
            all_faces.extend(result.faces)

        if not all_faces:
            print("No faces detected in any image.")
            return GroupingResult()

        print(f"  Detected {len(all_faces)} faces across all images")

        embeddings = self.embedder.get_embeddings_from_detections(all_faces)
        image_paths_per_face = [f.source_image_path or "" for f in all_faces]

        # 4. Cluster faces
        print("Step 3/3: Clustering faces by person...")
        groups = self.clusterer.cluster(embeddings, image_paths_per_face)

        # 5. Build result
        face_records = []
        for i, face in enumerate(all_faces):
            group_id = None
            for g in groups:
                if i in g.face_indices:
                    group_id = g.group_id
                    break

            record = FaceRecord(
                face_index=i,
                image_path=face.source_image_path or "",
                bbox=face.bbox,
                confidence=face.confidence,
                embedding=embeddings[i] if i < len(embeddings) else None,
                group_id=group_id,
            )
            face_records.append(record)

        result = GroupingResult(
            face_records=face_records,
            groups=groups,
            embeddings=embeddings,
        )

        print(f"\nResults: {result.num_faces} faces grouped into {result.num_groups} people")
        for group in groups:
            print(
                f"  Person {group.group_id}: {group.num_faces} faces "
                f"across {len(group.unique_images)} images"
            )

        # 6. Save results
        if output_dir:
            self._save_grouped_results(result, all_faces, output_dir)

        return result

    def find_person(
        self,
        query_image_path: str,
        search_dir: str,
        threshold: float = 0.6,
        show_progress: bool = True,
    ) -> Dict[str, float]:
        """Find all images containing a specific person.

        Args:
            query_image_path: Path to an image of the person to find.
            search_dir: Directory to search for matching images.
            threshold: Matching threshold (lower = stricter).
            show_progress: Whether to show progress bars.

        Returns:
            Dict mapping image paths to similarity scores.
        """
        # Detect and embed the query face
        query_result = self.detector.detect_from_file(query_image_path)
        if not query_result.faces:
            print(f"No face detected in query image: {query_image_path}")
            return {}

        # Use the largest face in the query image
        query_face = max(
            query_result.faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        query_embedding = self.embedder.get_embedding(query_face.face_crop)

        # Detect and embed all faces in the search directory
        image_paths = get_image_files(search_dir)
        if not image_paths:
            print(f"No images found in {search_dir}")
            return {}

        print(f"Searching {len(image_paths)} images for matching faces...")

        detection_results = self.detector.detect_batch(
            image_paths, show_progress=show_progress
        )

        all_faces = []
        face_image_paths = []
        for det_result in detection_results:
            for face in det_result.faces:
                all_faces.append(face)
                face_image_paths.append(det_result.image_path)

        if not all_faces:
            print("No faces found in search directory.")
            return {}

        all_embeddings = self.embedder.get_embeddings_from_detections(all_faces)

        # Find matches
        matching_images = self.clusterer.find_person_images(
            query_embedding, all_embeddings, face_image_paths, threshold
        )

        print(f"Found {len(matching_images)} images containing the person")
        return matching_images

    def _save_grouped_results(
        self,
        result: GroupingResult,
        faces: List[DetectedFace],
        output_dir: str,
    ) -> None:
        """Save grouped results to disk."""
        os.makedirs(output_dir, exist_ok=True)

        # Save JSON results
        result.save_json(os.path.join(output_dir, "grouping_results.json"))

        # Save face crops grouped by person
        for group in result.groups:
            group_dir = os.path.join(output_dir, f"person_{group.group_id}")
            os.makedirs(group_dir, exist_ok=True)

            for i, face_idx in enumerate(group.face_indices):
                if face_idx < len(faces) and faces[face_idx].face_crop is not None:
                    save_image(
                        faces[face_idx].face_crop,
                        os.path.join(group_dir, f"face_{i}.jpg"),
                    )

        # Save a summary of which images belong to which person
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Face Grouping Results\n")
            f.write(f"=====================\n\n")
            f.write(f"Total faces: {result.num_faces}\n")
            f.write(f"Total people: {result.num_groups}\n\n")

            for group in result.groups:
                f.write(f"Person {group.group_id} ({group.num_faces} faces):\n")
                for img_path in group.unique_images:
                    f.write(f"  - {img_path}\n")
                f.write("\n")

        print(f"Results saved to {output_dir}")
