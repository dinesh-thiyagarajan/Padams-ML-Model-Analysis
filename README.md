# Face Recognition & Grouping ML Model for Android

A machine learning pipeline that detects faces in images, extracts face embeddings, and groups photos by person — designed for integration into Android apps via TensorFlow Lite.

## How It Works

```
Images → Face Detection → Face Cropping → Embedding Extraction → Clustering → Groups by Person
```

1. **Face Detection**: Uses MTCNN to detect and crop faces from images
2. **Face Embedding**: Uses MobileFaceNet (optimized FaceNet) to generate 128-d embedding vectors for each face
3. **Face Clustering**: Uses DBSCAN/Agglomerative clustering to group embeddings by person identity
4. **Android Export**: Converts models to TFLite format for on-device inference

## Project Structure

```
├── face_recognition/
│   ├── detector.py          # Face detection (MTCNN)
│   ├── embedder.py          # Face embedding extraction (MobileFaceNet)
│   ├── clusterer.py         # Face clustering/grouping
│   ├── pipeline.py          # End-to-end pipeline
│   └── utils.py             # Utility functions
├── android_integration/
│   ├── FaceRecognitionHelper.kt   # Android helper class
│   └── FaceGroupingManager.kt     # Android grouping manager
├── scripts/
│   ├── train_embedder.py    # Training/fine-tuning script
│   ├── convert_to_tflite.py # TFLite conversion
│   └── demo.py              # Demo script
├── models/                  # Saved models (generated)
├── tests/
│   └── test_pipeline.py     # Unit tests
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Demo

```bash
# Group faces in a directory of images
python scripts/demo.py --input_dir path/to/your/images --output_dir output/

# Group all images containing a specific person
python scripts/demo.py --input_dir path/to/images --query_image path/to/my_face.jpg
```

### 3. Train/Fine-tune the Model

```bash
# Fine-tune on your own dataset (optional - pretrained works well)
python scripts/train_embedder.py --data_dir path/to/training/data --epochs 20
```

### 4. Export for Android

```bash
python scripts/convert_to_tflite.py --output models/face_embedder.tflite
```

## Android Integration

### Add TFLite dependency

```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
}
```

### Usage in Kotlin

```kotlin
val faceGrouping = FaceGroupingManager(context)

// Process all images from gallery
val groups = faceGrouping.groupFaces(imageUris)

// Find all images containing a specific person
val myPhotos = faceGrouping.findPersonInPhotos(referenceImage, allImages)
```

## Model Details

| Component | Model | Size | Input | Output |
|-----------|-------|------|-------|--------|
| Face Detection | MTCNN | ~2MB | Any image | Bounding boxes + landmarks |
| Face Embedding | MobileFaceNet | ~5MB | 112×112 face crop | 128-d embedding vector |

## Performance

- **Face Detection**: ~30ms per image on modern Android devices
- **Embedding Extraction**: ~15ms per face on modern Android devices
- **Clustering**: <1s for 1000 faces

## License

MIT
