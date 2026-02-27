# Android Integration Guide

## Step-by-step guide to integrate face grouping into your Android app.

### 1. Add Dependencies to `build.gradle` (app level)

```gradle
dependencies {
    // TensorFlow Lite for running the face embedding model
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'

    // Google ML Kit for face detection (runs on-device)
    implementation 'com.google.mlkit:face-detection:16.1.6'

    // Coroutines for async processing
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
}
```

### 2. Place the TFLite Model

Copy the generated `face_embedder.tflite` file to:
```
app/src/main/assets/face_embedder.tflite
```

### 3. Add Kotlin Files

Copy these files to your project:
- `FaceRecognitionHelper.kt` → Handles TFLite model inference
- `FaceGroupingManager.kt` → High-level face grouping API

### 4. Usage Examples

#### Group All Gallery Photos by Person

```kotlin
class GalleryActivity : AppCompatActivity() {

    private lateinit var faceGrouping: FaceGroupingManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        faceGrouping = FaceGroupingManager(this)
    }

    private fun groupPhotos() {
        lifecycleScope.launch {
            // Get all image URIs from gallery
            val imageUris = getGalleryImages()

            // Group faces — this runs in background
            val groups = faceGrouping.groupFaces(imageUris) { processed, total ->
                // Update progress bar
                progressBar.progress = (processed * 100) / total
            }

            // Display groups
            for (group in groups) {
                Log.d("FaceGroup", "Person ${group.groupId}: ${group.imageCount} photos")
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        faceGrouping.close()
    }
}
```

#### Find All Photos of a Specific Person

```kotlin
// User taps on a face in a photo
fun onFaceTapped(selectedFaceUri: Uri) {
    lifecycleScope.launch {
        val allPhotos = getGalleryImages()

        // Find all photos containing this person
        val matchingPhotos = faceGrouping.findPersonInPhotos(
            referenceFaceUri = selectedFaceUri,
            searchImageUris = allPhotos
        ) { processed, total ->
            updateProgress(processed, total)
        }

        // matchingPhotos is a list of (Uri, similarity) pairs
        // Display the matching photos
        displayResults(matchingPhotos.map { it.first })
    }
}
```

#### Compare Two Faces Directly

```kotlin
val helper = FaceRecognitionHelper(context)

// Get embeddings for two face crops
val embedding1 = helper.getEmbedding(faceBitmap1)
val embedding2 = helper.getEmbedding(faceBitmap2)

// Compare
val similarity = helper.compareFaces(embedding1, embedding2)
val isSame = helper.isSamePerson(embedding1, embedding2) // threshold = 0.5

Log.d("Face", "Similarity: $similarity, Same person: $isSame")
```

### 5. Performance Tips

- **Batch Processing**: Process images in the background using `WorkManager` for large galleries
- **Caching**: Store embeddings in a local database (Room) to avoid recomputing
- **Thumbnail Processing**: Use lower-resolution thumbnails for initial detection, full-res for embedding
- **Threading**: All heavy operations already run on `Dispatchers.IO` via coroutines

### 6. Caching Embeddings with Room (Recommended)

```kotlin
@Entity(tableName = "face_embeddings")
data class FaceEmbeddingEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val imageUri: String,
    val embedding: ByteArray,  // Serialize FloatArray to ByteArray
    val faceLeft: Int,
    val faceTop: Int,
    val faceRight: Int,
    val faceBottom: Int,
    val groupId: Int? = null,
)
```

This avoids re-processing images that have already been analyzed.
