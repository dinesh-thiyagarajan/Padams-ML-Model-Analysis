#!/usr/bin/env python3
"""Training and fine-tuning script for the MobileFaceNet face embedder.

This script trains the face embedding model using ArcFace loss, which is the
state-of-the-art loss function for face recognition. ArcFace adds an angular
margin penalty to improve intra-class compactness and inter-class separability.

Training data structure:
    data_dir/
        person_001/
            img1.jpg
            img2.jpg
            ...
        person_002/
            img1.jpg
            ...

Usage:
    # Train from scratch
    python scripts/train_embedder.py --data_dir path/to/training/data --epochs 50

    # Fine-tune an existing model
    python scripts/train_embedder.py --data_dir path/to/data --pretrained models/saved_model --epochs 20

    # Train with validation
    python scripts/train_embedder.py --data_dir path/to/train --val_dir path/to/val --epochs 50
"""

import argparse
import math
import os
import sys
from typing import Tuple

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_recognition.embedder import build_mobilefacenet


class ArcFaceLoss(tf.keras.layers.Layer):
    """ArcFace (Additive Angular Margin) loss layer.

    ArcFace applies an angular margin penalty in the normalized embedding
    space, which is geometrically interpretable and leads to highly
    discriminative face embeddings.

    The loss is: -log(e^(s*cos(theta_y + m)) / (e^(s*cos(theta_y + m)) + sum_j e^(s*cos(theta_j))))

    where:
        - theta_y is the angle between the feature and the ground truth class center
        - m is the angular margin (default 0.5 radians = ~28.6 degrees)
        - s is the feature scale (default 64)
    """

    def __init__(self, num_classes: int, margin: float = 0.5, scale: float = 64.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def build(self, input_shape):
        embedding_size = input_shape[-1]
        self.w = self.add_weight(
            name="arcface_weights",
            shape=(embedding_size, self.num_classes),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, embeddings, labels):
        # Normalize weights
        w_norm = tf.math.l2_normalize(self.w, axis=0)
        # embeddings are already L2-normalized from the model

        # Cosine similarity
        cosine = tf.matmul(embeddings, w_norm)
        cosine = tf.clip_by_value(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        # ArcFace angular margin
        sine = tf.sqrt(1.0 - tf.square(cosine))
        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Handle edge case when cos(theta) < cos(pi - m)
        phi = tf.where(cosine > self.threshold, phi, cosine - self.mm)

        # One-hot encode labels
        one_hot = tf.one_hot(labels, self.num_classes)

        # Apply margin only to the ground truth class
        logits = tf.where(tf.cast(one_hot, tf.bool), phi, cosine)
        logits = logits * self.scale

        return logits


def create_data_pipeline(
    data_dir: str,
    input_size: Tuple[int, int] = (112, 112),
    batch_size: int = 32,
    augment: bool = True,
) -> Tuple[tf.data.Dataset, int]:
    """Create a tf.data pipeline from a directory of person folders.

    Args:
        data_dir: Root directory with person sub-folders.
        input_size: Target image size.
        batch_size: Batch size.
        augment: Whether to apply data augmentation.

    Returns:
        Tuple of (dataset, num_classes).
    """
    # Get all person directories
    persons = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    num_classes = len(persons)
    person_to_label = {p: i for i, p in enumerate(persons)}

    # Collect all image paths and labels
    image_paths = []
    labels = []
    for person in persons:
        person_dir = os.path.join(data_dir, person)
        for fname in os.listdir(person_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in {".jpg", ".jpeg", ".png", ".bmp"}:
                image_paths.append(os.path.join(person_dir, fname))
                labels.append(person_to_label[person])

    print(f"Found {len(image_paths)} images of {num_classes} people")

    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, input_size)
        img = tf.cast(img, tf.float32)
        img = (img - 127.5) / 128.0  # Normalize to [-1, 1]
        return img, label

    def augment_fn(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.shuffle(len(image_paths))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, num_classes


class FaceTrainer:
    """Trainer for the face embedding model with ArcFace loss."""

    def __init__(
        self,
        embedding_size: int = 128,
        input_size: Tuple[int, int] = (112, 112),
        learning_rate: float = 0.01,
        pretrained_path: str = None,
    ):
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.pretrained_path = pretrained_path
        self.backbone = None
        self.training_model = None

    def build_training_model(self, num_classes: int) -> tf.keras.Model:
        """Build the full training model with ArcFace head.

        The training model consists of:
        1. MobileFaceNet backbone → 128-d embeddings
        2. ArcFace loss layer → class logits

        At inference time, only the backbone is used.
        """
        if self.pretrained_path and os.path.exists(self.pretrained_path):
            print(f"Loading pretrained model from {self.pretrained_path}")
            self.backbone = tf.keras.models.load_model(self.pretrained_path)
        else:
            self.backbone = build_mobilefacenet(
                input_shape=(*self.input_size, 3),
                embedding_size=self.embedding_size,
            )

        # Training model with ArcFace head
        face_input = self.backbone.input
        embeddings = self.backbone.output

        label_input = tf.keras.Input(shape=(), dtype=tf.int32, name="label_input")

        arcface = ArcFaceLoss(num_classes=num_classes, margin=0.5, scale=64.0)
        logits = arcface(embeddings, label_input)

        self.training_model = tf.keras.Model(
            inputs=[face_input, label_input],
            outputs=logits,
            name="FaceTrainingModel",
        )

        return self.training_model

    def train(
        self,
        train_dataset: tf.data.Dataset,
        num_classes: int,
        epochs: int = 50,
        val_dataset: tf.data.Dataset = None,
        output_dir: str = "models",
    ):
        """Train the face embedding model.

        Args:
            train_dataset: Training dataset yielding (images, labels).
            num_classes: Number of unique people in the training set.
            epochs: Number of training epochs.
            val_dataset: Optional validation dataset.
            output_dir: Directory to save the trained model.
        """
        self.build_training_model(num_classes)

        # Learning rate schedule: warmup + cosine decay
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=epochs * 100,  # Approximate steps
            alpha=1e-6,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.training_model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=["accuracy"],
        )

        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(output_dir, "best_model.keras"),
                monitor="loss",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="loss",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="loss",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1,
            ),
        ]

        # Wrap dataset to provide (images, labels) as inputs and labels as targets
        def wrap_dataset(ds):
            return ds.map(lambda img, lbl: ({"face_input": img, "label_input": lbl}, lbl))

        train_ds = wrap_dataset(train_dataset)
        val_ds = wrap_dataset(val_dataset) if val_dataset else None

        print(f"\nStarting training for {epochs} epochs...")
        print(f"  Number of classes: {num_classes}")
        print(f"  Embedding size: {self.embedding_size}")
        print(f"  Input size: {self.input_size}")
        print(f"  Learning rate: {self.learning_rate}")

        history = self.training_model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
        )

        # Save the backbone (embedding model) for inference
        os.makedirs(output_dir, exist_ok=True)
        backbone_path = os.path.join(output_dir, "face_embedder")
        self.backbone.save(backbone_path)
        print(f"\nBackbone model saved to: {backbone_path}")

        return history

    def evaluate_embeddings(
        self,
        dataset: tf.data.Dataset,
    ) -> dict:
        """Evaluate the quality of learned embeddings.

        Computes:
        - Average intra-class distance (should be small)
        - Average inter-class distance (should be large)
        - Separation ratio (inter / intra, should be large)
        """
        all_embeddings = []
        all_labels = []

        for images, labels in dataset:
            embeddings = self.backbone.predict(images, verbose=0)
            all_embeddings.append(embeddings)
            all_labels.append(labels.numpy())

        all_embeddings = np.concatenate(all_embeddings)
        all_labels = np.concatenate(all_labels)

        unique_labels = np.unique(all_labels)

        intra_distances = []
        centroids = {}

        for label in unique_labels:
            mask = all_labels == label
            class_embeddings = all_embeddings[mask]

            if len(class_embeddings) < 2:
                continue

            centroid = class_embeddings.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            centroids[label] = centroid

            # Intra-class distances
            for emb in class_embeddings:
                dist = 1.0 - np.dot(emb, centroid)
                intra_distances.append(dist)

        # Inter-class distances
        inter_distances = []
        centroid_list = list(centroids.values())
        for i in range(len(centroid_list)):
            for j in range(i + 1, len(centroid_list)):
                dist = 1.0 - np.dot(centroid_list[i], centroid_list[j])
                inter_distances.append(dist)

        avg_intra = np.mean(intra_distances) if intra_distances else 0
        avg_inter = np.mean(inter_distances) if inter_distances else 0
        ratio = avg_inter / avg_intra if avg_intra > 0 else float("inf")

        metrics = {
            "avg_intra_class_distance": float(avg_intra),
            "avg_inter_class_distance": float(avg_inter),
            "separation_ratio": float(ratio),
            "num_classes": len(unique_labels),
            "num_embeddings": len(all_embeddings),
        }

        print("\nEmbedding Quality Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        return metrics


def main():
    parser = argparse.ArgumentParser(description="Train MobileFaceNet face embedder")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Training data directory (person sub-folders with images)",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default=None,
        help="Validation data directory (same structure as data_dir)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained model for fine-tuning",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--embedding_size", type=int, default=128, help="Embedding vector size"
    )
    parser.add_argument(
        "--output_dir", type=str, default="models", help="Output directory for saved models"
    )
    args = parser.parse_args()

    # Create datasets
    train_dataset, num_classes = create_data_pipeline(
        args.data_dir, batch_size=args.batch_size, augment=True
    )

    val_dataset = None
    if args.val_dir:
        val_dataset, _ = create_data_pipeline(
            args.val_dir, batch_size=args.batch_size, augment=False
        )

    # Train
    trainer = FaceTrainer(
        embedding_size=args.embedding_size,
        learning_rate=args.lr,
        pretrained_path=args.pretrained,
    )
    trainer.train(
        train_dataset=train_dataset,
        num_classes=num_classes,
        epochs=args.epochs,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
    )

    # Evaluate
    trainer.evaluate_embeddings(train_dataset)


if __name__ == "__main__":
    main()
