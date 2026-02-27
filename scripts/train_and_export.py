#!/usr/bin/env python3
"""Simplified training + export script that trains the model and saves it."""

import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_recognition.embedder import build_mobilefacenet
from scripts.train_embedder import ArcFaceLoss, create_data_pipeline


def train_and_export(
    data_dir: str,
    output_dir: str = "models",
    epochs: int = 15,
    batch_size: int = 16,
    embedding_size: int = 128,
    learning_rate: float = 0.01,
):
    """Train the face embedding model and export for inference."""

    os.makedirs(output_dir, exist_ok=True)

    # Create dataset
    train_dataset, num_classes = create_data_pipeline(
        data_dir, batch_size=batch_size, augment=True
    )

    # Build backbone
    print("Building MobileFaceNet backbone...")
    backbone = build_mobilefacenet(
        input_shape=(112, 112, 3),
        embedding_size=embedding_size,
    )
    backbone.summary(print_fn=lambda x: None)  # Suppress verbose summary
    print(f"  Parameters: {backbone.count_params():,}")

    # Build training model with ArcFace head
    face_input = backbone.input
    embeddings = backbone.output

    label_input = tf.keras.Input(shape=(), dtype=tf.int32, name="label_input")
    arcface = ArcFaceLoss(num_classes=num_classes, margin=0.5, scale=64.0)
    logits = arcface(embeddings, label_input)

    training_model = tf.keras.Model(
        inputs=[face_input, label_input],
        outputs=logits,
        name="FaceTrainingModel",
    )

    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    training_model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    # Wrap dataset
    def wrap_dataset(ds):
        return ds.map(lambda img, lbl: ({"face_input": img, "label_input": lbl}, lbl))

    train_ds = wrap_dataset(train_dataset)

    # Train
    print(f"\nTraining for {epochs} epochs on {num_classes} classes...")
    history = training_model.fit(train_ds, epochs=epochs, verbose=1)

    final_acc = history.history["accuracy"][-1]
    final_loss = history.history["loss"][-1]
    print(f"\nTraining complete!")
    print(f"  Final accuracy: {final_acc:.4f}")
    print(f"  Final loss: {final_loss:.4f}")

    # Save the backbone (inference model)
    backbone_path = os.path.join(output_dir, "face_embedder_trained.keras")
    backbone.save(backbone_path)
    print(f"\nBackbone saved to: {backbone_path}")

    # Evaluate embedding quality
    print("\nEvaluating embedding quality...")
    all_embeddings = []
    all_labels = []
    for images, labels in train_dataset:
        embs = backbone.predict(images, verbose=0)
        all_embeddings.append(embs)
        all_labels.append(labels.numpy())

    all_embeddings = np.concatenate(all_embeddings)
    all_labels = np.concatenate(all_labels)

    unique_labels = np.unique(all_labels)
    centroids = {}
    intra_distances = []

    for label in unique_labels:
        mask = all_labels == label
        class_embs = all_embeddings[mask]
        if len(class_embs) < 2:
            continue
        centroid = class_embs.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids[label] = centroid
        for emb in class_embs:
            intra_distances.append(1.0 - np.dot(emb, centroid))

    inter_distances = []
    centroid_list = list(centroids.values())
    for i in range(len(centroid_list)):
        for j in range(i + 1, len(centroid_list)):
            inter_distances.append(1.0 - np.dot(centroid_list[i], centroid_list[j]))

    avg_intra = np.mean(intra_distances)
    avg_inter = np.mean(inter_distances)
    ratio = avg_inter / avg_intra if avg_intra > 0 else float("inf")

    print(f"  Avg intra-class distance: {avg_intra:.4f} (lower = tighter clusters)")
    print(f"  Avg inter-class distance: {avg_inter:.4f} (higher = better separation)")
    print(f"  Separation ratio: {ratio:.2f}x (higher = better)")

    # Convert to TFLite
    print("\nConverting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(backbone)
    tflite_model = converter.convert()

    tflite_path = os.path.join(output_dir, "face_embedder.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"  TFLite model saved to: {tflite_path}")
    print(f"  Model size: {size_mb:.2f} MB")

    # Verify TFLite model
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    test_input = np.random.randn(1, 112, 112, 3).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], test_input)
    interpreter.invoke()
    test_output = interpreter.get_tensor(output_details[0]["index"])
    norm = np.linalg.norm(test_output)
    print(f"  TFLite output shape: {test_output.shape}")
    print(f"  TFLite output norm: {norm:.4f} (should be ~1.0)")

    # Save training history
    history_path = os.path.join(output_dir, "training_history.txt")
    with open(history_path, "w") as f:
        f.write("Epoch\tLoss\tAccuracy\n")
        for i in range(len(history.history["loss"])):
            f.write(f"{i+1}\t{history.history['loss'][i]:.4f}\t{history.history['accuracy'][i]:.4f}\n")
        f.write(f"\nFinal metrics:\n")
        f.write(f"  Accuracy: {final_acc:.4f}\n")
        f.write(f"  Loss: {final_loss:.4f}\n")
        f.write(f"  Intra-class distance: {avg_intra:.4f}\n")
        f.write(f"  Inter-class distance: {avg_inter:.4f}\n")
        f.write(f"  Separation ratio: {ratio:.2f}x\n")
        f.write(f"  TFLite model size: {size_mb:.2f} MB\n")

    print(f"\n{'='*50}")
    print(f"All done! Files saved to {output_dir}/")
    print(f"  face_embedder_trained.keras  - Keras model for further training")
    print(f"  face_embedder.tflite         - TFLite model for Android")
    print(f"  training_history.txt         - Training metrics log")
    print(f"{'='*50}")


if __name__ == "__main__":
    train_and_export(
        data_dir="data/synthetic_faces",
        output_dir="models",
        epochs=15,
        batch_size=16,
    )
