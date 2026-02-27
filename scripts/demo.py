#!/usr/bin/env python3
"""Demo script for the face grouping pipeline.

Examples:
    # Group all faces in a directory of photos
    python scripts/demo.py --input_dir path/to/photos --output_dir output/

    # Find all photos containing a specific person
    python scripts/demo.py --input_dir path/to/photos --query_image path/to/my_face.jpg

    # Use a trained model
    python scripts/demo.py --input_dir path/to/photos --model models/face_embedder --output_dir output/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_recognition.pipeline import FaceGroupingPipeline


def demo_group_faces(input_dir: str, output_dir: str, model_path: str = None):
    """Demo: Group all faces in a directory by person."""
    print("=" * 60)
    print("Face Grouping Demo")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    if model_path:
        print(f"Model: {model_path}")
    print()

    pipeline = FaceGroupingPipeline(model_path=model_path)
    result = pipeline.group_faces_in_directory(input_dir, output_dir)

    print(f"\n{'=' * 60}")
    print(f"Grouping Complete!")
    print(f"{'=' * 60}")
    print(f"Total faces detected: {result.num_faces}")
    print(f"Total people found: {result.num_groups}")

    for group in result.groups:
        print(f"\n  Person {group.group_id}:")
        print(f"    Faces: {group.num_faces}")
        print(f"    Images: {len(group.unique_images)}")
        for img_path in group.unique_images:
            print(f"      - {os.path.basename(img_path)}")

    if output_dir:
        print(f"\nResults saved to: {output_dir}")
        print(f"  - grouping_results.json: Machine-readable results")
        print(f"  - summary.txt: Human-readable summary")
        print(f"  - person_*/: Face crops grouped by person")

    return result


def demo_find_person(
    input_dir: str,
    query_image: str,
    model_path: str = None,
    threshold: float = 0.6,
):
    """Demo: Find all images containing a specific person."""
    print("=" * 60)
    print("Find Person Demo")
    print("=" * 60)
    print(f"Query image: {query_image}")
    print(f"Search directory: {input_dir}")
    if model_path:
        print(f"Model: {model_path}")
    print()

    pipeline = FaceGroupingPipeline(model_path=model_path)
    matches = pipeline.find_person(query_image, input_dir, threshold=threshold)

    print(f"\n{'=' * 60}")
    print(f"Search Complete!")
    print(f"{'=' * 60}")
    print(f"Found {len(matches)} matching images:\n")

    for image_path, similarity in matches.items():
        print(f"  {os.path.basename(image_path):40s} (similarity: {similarity:.3f})")

    return matches


def main():
    parser = argparse.ArgumentParser(description="Face Grouping Demo")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing images to process",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save grouped results",
    )
    parser.add_argument(
        "--query_image",
        type=str,
        default=None,
        help="Path to a face image to search for (enables find-person mode)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (Keras or TFLite)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Matching threshold (lower = stricter, default 0.6)",
    )
    args = parser.parse_args()

    if args.query_image:
        demo_find_person(args.input_dir, args.query_image, args.model, args.threshold)
    else:
        if not args.output_dir:
            args.output_dir = os.path.join(args.input_dir, "face_groups_output")
        demo_group_faces(args.input_dir, args.output_dir, args.model)


if __name__ == "__main__":
    main()
