#!/usr/bin/env python3
"""Convert the MobileFaceNet model to TFLite format for Android deployment.

Usage:
    python scripts/convert_to_tflite.py --output models/face_embedder.tflite
    python scripts/convert_to_tflite.py --model_path models/saved_model --output models/face_embedder.tflite
    python scripts/convert_to_tflite.py --output models/face_embedder.tflite --quantize
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_recognition.embedder import build_mobilefacenet


def convert_to_tflite(
    model_path: str = None,
    output_path: str = "models/face_embedder.tflite",
    quantize: bool = False,
    input_size: tuple = (112, 112),
    embedding_size: int = 128,
) -> str:
    """Convert a Keras model to TFLite format.

    Args:
        model_path: Path to saved Keras model. If None, builds a fresh model.
        output_path: Path to save the TFLite model.
        quantize: Whether to apply dynamic range quantization.
        input_size: Input image size (H, W).
        embedding_size: Embedding vector size.

    Returns:
        Path to the saved TFLite model.
    """
    print("Loading model...")
    if model_path and os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        print("No saved model found. Building fresh MobileFaceNet...")
        model = build_mobilefacenet(
            input_shape=(*input_size, 3),
            embedding_size=embedding_size,
        )

    model.summary()

    # Convert to TFLite
    print("\nConverting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        print("Applying dynamic range quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Representative dataset for full integer quantization
        def representative_dataset():
            for _ in range(100):
                data = np.random.randn(1, *input_size, 3).astype(np.float32)
                yield [data]

        converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    model_size_mb = len(tflite_model) / (1024 * 1024)
    print(f"\nTFLite model saved to: {output_path}")
    print(f"Model size: {model_size_mb:.2f} MB")

    # Verify the converted model
    print("\nVerifying TFLite model...")
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output dtype: {output_details[0]['dtype']}")

    # Test inference
    test_input = np.random.randn(1, *input_size, 3).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], test_input)
    interpreter.invoke()
    test_output = interpreter.get_tensor(output_details[0]["index"])
    print(f"Test output shape: {test_output.shape}")
    print(f"Test output norm: {np.linalg.norm(test_output):.4f} (should be ~1.0)")

    print("\nConversion successful!")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert MobileFaceNet to TFLite for Android"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to saved Keras model directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/face_embedder.tflite",
        help="Output TFLite file path",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply dynamic range quantization (smaller model, slight accuracy loss)",
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=128,
        help="Embedding vector size",
    )
    args = parser.parse_args()

    convert_to_tflite(
        model_path=args.model_path,
        output_path=args.output,
        quantize=args.quantize,
        embedding_size=args.embedding_size,
    )


if __name__ == "__main__":
    main()
