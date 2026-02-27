"""Unit tests for the face recognition pipeline components."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_recognition.clusterer import FaceClusterer
from face_recognition.embedder import build_mobilefacenet
from face_recognition.utils import cosine_similarity, l2_distance, preprocess_face


class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def test_cosine_similarity_identical(self):
        a = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(a, a), 1.0, places=5)

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0, places=5)

    def test_cosine_similarity_opposite(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(a, b), -1.0, places=5)

    def test_l2_distance_same(self):
        a = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(l2_distance(a, a), 0.0, places=5)

    def test_l2_distance_known(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        self.assertAlmostEqual(l2_distance(a, b), 5.0, places=5)

    def test_preprocess_face_shape(self):
        face = np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8)
        result = preprocess_face(face, target_size=(112, 112))
        self.assertEqual(result.shape, (112, 112, 3))
        self.assertEqual(result.dtype, np.float32)

    def test_preprocess_face_range(self):
        # All zeros -> (0 - 127.5) / 128 = ~ -0.996
        face = np.zeros((100, 100, 3), dtype=np.uint8)
        result = preprocess_face(face)
        self.assertTrue(np.all(result >= -1.0))
        self.assertTrue(np.all(result <= 1.0))

        # All 255 -> (255 - 127.5) / 128 = ~ 0.996
        face = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = preprocess_face(face)
        self.assertTrue(np.all(result >= -1.0))
        self.assertTrue(np.all(result <= 1.0))


class TestMobileFaceNet(unittest.TestCase):
    """Test the MobileFaceNet model architecture."""

    def test_model_builds(self):
        model = build_mobilefacenet(input_shape=(112, 112, 3), embedding_size=128)
        self.assertIsNotNone(model)

    def test_model_output_shape(self):
        model = build_mobilefacenet(input_shape=(112, 112, 3), embedding_size=128)
        dummy_input = np.random.randn(1, 112, 112, 3).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)
        self.assertEqual(output.shape, (1, 128))

    def test_model_output_normalized(self):
        model = build_mobilefacenet(input_shape=(112, 112, 3), embedding_size=128)
        dummy_input = np.random.randn(2, 112, 112, 3).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)

        for i in range(output.shape[0]):
            norm = np.linalg.norm(output[i])
            self.assertAlmostEqual(norm, 1.0, places=3)

    def test_model_batch_inference(self):
        model = build_mobilefacenet(input_shape=(112, 112, 3), embedding_size=128)
        batch = np.random.randn(4, 112, 112, 3).astype(np.float32)
        output = model.predict(batch, verbose=0)
        self.assertEqual(output.shape, (4, 128))


class TestFaceClusterer(unittest.TestCase):
    """Test the face clustering module."""

    def _make_embeddings(self, n_people: int, n_per_person: int, noise: float = 0.05):
        """Generate synthetic face embeddings for testing."""
        rng = np.random.RandomState(42)
        embeddings = []
        labels = []

        for person_id in range(n_people):
            # Random center for this person
            center = rng.randn(128)
            center = center / np.linalg.norm(center)

            for _ in range(n_per_person):
                emb = center + rng.randn(128) * noise
                emb = emb / np.linalg.norm(emb)
                embeddings.append(emb)
                labels.append(person_id)

        return np.array(embeddings), labels

    def test_cluster_single_person(self):
        embeddings, _ = self._make_embeddings(1, 5)
        clusterer = FaceClusterer(method="dbscan", distance_threshold=0.6, min_samples=2)
        groups = clusterer.cluster(embeddings)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].num_faces, 5)

    def test_cluster_multiple_people(self):
        embeddings, _ = self._make_embeddings(3, 5)
        clusterer = FaceClusterer(method="dbscan", distance_threshold=0.3, min_samples=2)
        groups = clusterer.cluster(embeddings)
        # Should find approximately 3 groups
        self.assertGreaterEqual(len(groups), 2)
        self.assertLessEqual(len(groups), 5)

    def test_cluster_with_image_paths(self):
        embeddings, labels = self._make_embeddings(2, 3)
        paths = [f"img_{i}.jpg" for i in range(len(embeddings))]
        clusterer = FaceClusterer(method="dbscan", distance_threshold=0.3, min_samples=2)
        groups = clusterer.cluster(embeddings, image_paths=paths)

        for group in groups:
            self.assertTrue(len(group.image_paths) > 0)
            self.assertIsNotNone(group.representative_index)

    def test_find_matching_faces(self):
        embeddings, _ = self._make_embeddings(3, 5)
        clusterer = FaceClusterer()

        # Query with the first embedding (should match first 5)
        matches = clusterer.find_matching_faces(
            embeddings[0], embeddings, threshold=0.3
        )
        self.assertTrue(len(matches) > 0)
        # First match should be itself
        self.assertEqual(matches[0][0], 0)
        self.assertGreater(matches[0][1], 0.9)

    def test_find_person_images(self):
        embeddings, _ = self._make_embeddings(2, 3)
        paths = [f"person{i // 3}_img{i % 3}.jpg" for i in range(6)]
        clusterer = FaceClusterer()

        result = clusterer.find_person_images(
            embeddings[0], embeddings, paths, threshold=0.3
        )
        self.assertIsInstance(result, dict)
        self.assertTrue(len(result) > 0)

    def test_empty_embeddings(self):
        clusterer = FaceClusterer()
        groups = clusterer.cluster(np.array([]).reshape(0, 128))
        self.assertEqual(len(groups), 0)

    def test_single_embedding(self):
        embedding = np.random.randn(1, 128).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        clusterer = FaceClusterer()
        groups = clusterer.cluster(embedding, image_paths=["test.jpg"])
        self.assertEqual(len(groups), 1)

    def test_agglomerative_clustering(self):
        embeddings, _ = self._make_embeddings(3, 5)
        clusterer = FaceClusterer(
            method="agglomerative",
            n_clusters=3,
        )
        groups = clusterer.cluster(embeddings)
        self.assertEqual(len(groups), 3)


if __name__ == "__main__":
    unittest.main()
