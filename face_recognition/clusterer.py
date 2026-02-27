"""Face clustering/grouping module.

Groups face embeddings into clusters where each cluster represents a unique
person. Supports both unsupervised clustering (discover all people) and
query-based grouping (find all images of a specific person).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

from face_recognition.utils import cosine_similarity


@dataclass
class FaceGroup:
    """A group of faces belonging to the same person."""

    group_id: int
    face_indices: List[int] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)
    representative_index: Optional[int] = None  # Index of the most central face

    @property
    def num_faces(self) -> int:
        return len(self.face_indices)

    @property
    def unique_images(self) -> List[str]:
        return list(set(self.image_paths))


class FaceClusterer:
    """Groups face embeddings into clusters by person identity.

    Uses DBSCAN by default, which automatically determines the number of
    clusters (people) and handles outliers (unrecognized faces). Also supports
    Agglomerative Clustering when the number of people is known.
    """

    def __init__(
        self,
        method: str = "dbscan",
        distance_threshold: float = 0.6,
        min_samples: int = 2,
        n_clusters: Optional[int] = None,
    ):
        """
        Args:
            method: Clustering method - "dbscan" or "agglomerative".
            distance_threshold: Maximum cosine distance between faces of the
                same person. Lower = stricter matching. Default 0.6 works well.
            min_samples: Minimum faces to form a group (DBSCAN only).
            n_clusters: Number of people (agglomerative only). If None, uses
                distance_threshold to determine clusters.
        """
        self.method = method
        self.distance_threshold = distance_threshold
        self.min_samples = min_samples
        self.n_clusters = n_clusters

    def cluster(
        self,
        embeddings: np.ndarray,
        image_paths: Optional[List[str]] = None,
    ) -> List[FaceGroup]:
        """Cluster face embeddings into groups.

        Args:
            embeddings: Array of shape (N, embedding_dim) with face embeddings.
            image_paths: Optional list of source image paths for each face.

        Returns:
            List of FaceGroup, each representing a person.
        """
        if len(embeddings) == 0:
            return []

        if len(embeddings) == 1:
            paths = [image_paths[0]] if image_paths else []
            return [FaceGroup(group_id=0, face_indices=[0], image_paths=paths)]

        # Compute cosine distance matrix
        distance_matrix = cosine_distances(embeddings)

        if self.method == "dbscan":
            labels = self._cluster_dbscan(distance_matrix)
        elif self.method == "agglomerative":
            labels = self._cluster_agglomerative(distance_matrix)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        return self._labels_to_groups(labels, embeddings, image_paths)

    def _cluster_dbscan(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Cluster using DBSCAN."""
        clusterer = DBSCAN(
            eps=self.distance_threshold,
            min_samples=self.min_samples,
            metric="precomputed",
        )
        return clusterer.fit_predict(distance_matrix)

    def _cluster_agglomerative(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Cluster using Agglomerative Clustering."""
        if self.n_clusters is not None:
            clusterer = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                metric="precomputed",
                linkage="average",
            )
        else:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.distance_threshold,
                metric="precomputed",
                linkage="average",
            )
        return clusterer.fit_predict(distance_matrix)

    def _labels_to_groups(
        self,
        labels: np.ndarray,
        embeddings: np.ndarray,
        image_paths: Optional[List[str]],
    ) -> List[FaceGroup]:
        """Convert cluster labels to FaceGroup objects."""
        groups = {}

        for idx, label in enumerate(labels):
            if label == -1:
                # DBSCAN noise point — create a singleton group
                noise_id = max(groups.keys(), default=-1) + 1
                while noise_id in groups:
                    noise_id += 1
                paths = [image_paths[idx]] if image_paths else []
                groups[noise_id] = FaceGroup(
                    group_id=noise_id,
                    face_indices=[idx],
                    image_paths=paths,
                )
                continue

            if label not in groups:
                groups[label] = FaceGroup(group_id=int(label))

            groups[label].face_indices.append(idx)
            if image_paths:
                groups[label].image_paths.append(image_paths[idx])

        # Find representative face for each group (closest to centroid)
        for group in groups.values():
            if len(group.face_indices) > 1:
                group_embeddings = embeddings[group.face_indices]
                centroid = group_embeddings.mean(axis=0)
                centroid = centroid / np.linalg.norm(centroid)

                similarities = group_embeddings @ centroid
                best_local = int(np.argmax(similarities))
                group.representative_index = group.face_indices[best_local]
            else:
                group.representative_index = group.face_indices[0]

        # Sort groups by size (largest first)
        sorted_groups = sorted(groups.values(), key=lambda g: g.num_faces, reverse=True)

        # Re-assign group IDs
        for i, group in enumerate(sorted_groups):
            group.group_id = i

        return sorted_groups

    def find_matching_faces(
        self,
        query_embedding: np.ndarray,
        all_embeddings: np.ndarray,
        threshold: float = 0.6,
    ) -> List[Tuple[int, float]]:
        """Find all faces matching a query face.

        This is the core "find all photos of me" functionality. Given a
        reference face embedding, it finds all other face embeddings that
        are similar enough to be the same person.

        Args:
            query_embedding: 128-d embedding of the query face.
            all_embeddings: Array of shape (N, 128) with all face embeddings.
            threshold: Maximum cosine distance for a match.

        Returns:
            List of (face_index, similarity_score) tuples, sorted by
            similarity (most similar first).
        """
        if len(all_embeddings) == 0:
            return []

        query = query_embedding.reshape(1, -1)
        distances = cosine_distances(query, all_embeddings)[0]

        matches = []
        for idx, dist in enumerate(distances):
            if dist < threshold:
                similarity = 1.0 - dist
                matches.append((idx, float(similarity)))

        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def find_person_images(
        self,
        query_embedding: np.ndarray,
        all_embeddings: np.ndarray,
        image_paths: List[str],
        threshold: float = 0.6,
    ) -> Dict[str, float]:
        """Find all images containing a specific person.

        Args:
            query_embedding: Embedding of the person to search for.
            all_embeddings: All face embeddings from all images.
            image_paths: Source image path for each embedding.
            threshold: Matching threshold.

        Returns:
            Dict mapping image_path to best match similarity score.
        """
        matches = self.find_matching_faces(query_embedding, all_embeddings, threshold)

        image_scores = {}
        for face_idx, similarity in matches:
            path = image_paths[face_idx]
            if path not in image_scores or similarity > image_scores[path]:
                image_scores[path] = similarity

        return dict(sorted(image_scores.items(), key=lambda x: x[1], reverse=True))
