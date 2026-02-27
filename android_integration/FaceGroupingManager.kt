package com.example.facerecognition

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.net.Uri
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.sqrt

/**
 * Face data class representing a detected face with its embedding.
 */
data class FaceData(
    val imageUri: Uri,
    val faceRect: Rect,
    val embedding: FloatArray,
    val confidence: Float,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is FaceData) return false
        return imageUri == other.imageUri && faceRect == other.faceRect
    }

    override fun hashCode(): Int {
        return 31 * imageUri.hashCode() + faceRect.hashCode()
    }
}

/**
 * A group of faces belonging to the same person.
 */
data class FaceGroupResult(
    val groupId: Int,
    val faces: List<FaceData>,
    val representativeFace: FaceData,
) {
    /** All unique image URIs in this group. */
    val imageUris: List<Uri>
        get() = faces.map { it.imageUri }.distinct()

    /** Number of images this person appears in. */
    val imageCount: Int
        get() = imageUris.size
}

/**
 * High-level manager for face grouping on Android.
 *
 * Combines Google ML Kit for face detection with a custom TFLite model for
 * face embedding extraction. Groups photos by person using clustering.
 *
 * Usage:
 * ```kotlin
 * val manager = FaceGroupingManager(context)
 *
 * // Group all photos by person
 * val groups = manager.groupFaces(imageUris)
 *
 * // Find all photos of a specific person
 * val myPhotos = manager.findPersonInPhotos(myFaceUri, allImageUris)
 *
 * manager.close()
 * ```
 */
class FaceGroupingManager(
    private val context: Context,
    modelFileName: String = "face_embedder.tflite",
    private val similarityThreshold: Float = 0.5f,
) {
    private val faceHelper = FaceRecognitionHelper(context, modelFileName)

    // ML Kit face detector options
    private val detectorOptions = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
        .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
        .setMinFaceSize(0.15f)
        .build()

    private val faceDetector = FaceDetection.getClient(detectorOptions)

    /**
     * Detect faces in an image and extract embeddings.
     *
     * @param imageUri URI of the image to process
     * @return List of FaceData for each detected face
     */
    suspend fun detectAndEmbedFaces(imageUri: Uri): List<FaceData> = withContext(Dispatchers.IO) {
        val bitmap = loadBitmap(imageUri) ?: return@withContext emptyList()

        val faces = detectFaces(bitmap, imageUri)
        val faceDataList = mutableListOf<FaceData>()

        for ((rect, confidence) in faces) {
            val faceCrop = faceHelper.cropFace(bitmap, rect)
            val embedding = faceHelper.getEmbedding(faceCrop)
            faceCrop.recycle()

            faceDataList.add(
                FaceData(
                    imageUri = imageUri,
                    faceRect = rect,
                    embedding = embedding,
                    confidence = confidence,
                )
            )
        }

        bitmap.recycle()
        faceDataList
    }

    /**
     * Detect faces using Google ML Kit.
     */
    private suspend fun detectFaces(
        bitmap: Bitmap,
        imageUri: Uri,
    ): List<Pair<Rect, Float>> = suspendCancellableCoroutine { continuation ->
        val inputImage = InputImage.fromBitmap(bitmap, 0)

        faceDetector.process(inputImage)
            .addOnSuccessListener { faces ->
                val results = faces.map { face ->
                    Pair(face.boundingBox, face.trackingId?.toFloat() ?: 1.0f)
                }
                continuation.resume(results)
            }
            .addOnFailureListener { e ->
                continuation.resumeWithException(e)
            }
    }

    /**
     * Group faces from multiple images by person identity.
     *
     * This is the main "group all photos by person" feature, similar to
     * Google Photos' face grouping.
     *
     * @param imageUris List of image URIs to process
     * @param onProgress Callback with (processed, total) for progress tracking
     * @return List of FaceGroupResult, sorted by group size (largest first)
     */
    suspend fun groupFaces(
        imageUris: List<Uri>,
        onProgress: ((Int, Int) -> Unit)? = null,
    ): List<FaceGroupResult> = withContext(Dispatchers.Default) {
        // Step 1: Detect faces and extract embeddings from all images
        val allFaces = mutableListOf<FaceData>()

        for ((index, uri) in imageUris.withIndex()) {
            try {
                val faces = detectAndEmbedFaces(uri)
                allFaces.addAll(faces)
            } catch (e: Exception) {
                // Skip images that fail to process
            }
            onProgress?.invoke(index + 1, imageUris.size)
        }

        if (allFaces.isEmpty()) return@withContext emptyList()

        // Step 2: Cluster faces using greedy approach
        clusterFaces(allFaces)
    }

    /**
     * Find all images containing a specific person.
     *
     * @param referenceFaceUri URI of an image containing the person to find
     * @param searchImageUris URIs of images to search through
     * @param onProgress Progress callback
     * @return List of matching image URIs with similarity scores, sorted by similarity
     */
    suspend fun findPersonInPhotos(
        referenceFaceUri: Uri,
        searchImageUris: List<Uri>,
        onProgress: ((Int, Int) -> Unit)? = null,
    ): List<Pair<Uri, Float>> = withContext(Dispatchers.Default) {
        // Get the reference face embedding
        val referenceFaces = detectAndEmbedFaces(referenceFaceUri)
        if (referenceFaces.isEmpty()) return@withContext emptyList()

        // Use the largest face as reference
        val referenceEmbedding = referenceFaces.maxByOrNull { face ->
            face.faceRect.width() * face.faceRect.height()
        }?.embedding ?: return@withContext emptyList()

        // Search all images
        val matches = mutableListOf<Pair<Uri, Float>>()

        for ((index, uri) in searchImageUris.withIndex()) {
            try {
                val faces = detectAndEmbedFaces(uri)
                for (face in faces) {
                    val similarity = faceHelper.compareFaces(referenceEmbedding, face.embedding)
                    if (similarity > similarityThreshold) {
                        matches.add(Pair(uri, similarity))
                        break // Found a match in this image, move to next
                    }
                }
            } catch (e: Exception) {
                // Skip failed images
            }
            onProgress?.invoke(index + 1, searchImageUris.size)
        }

        matches.sortedByDescending { it.second }
    }

    /**
     * Cluster faces into groups using greedy agglomerative approach.
     *
     * This is a simple but effective on-device clustering algorithm:
     * 1. Start with each face as its own cluster
     * 2. For each face, find the most similar existing cluster
     * 3. If similarity > threshold, add to that cluster; otherwise create new one
     */
    private fun clusterFaces(faces: List<FaceData>): List<FaceGroupResult> {
        if (faces.isEmpty()) return emptyList()

        // Greedy clustering
        val clusters = mutableListOf<MutableList<FaceData>>()

        for (face in faces) {
            var bestClusterIdx = -1
            var bestSimilarity = similarityThreshold

            for ((clusterIdx, cluster) in clusters.withIndex()) {
                // Compare with the representative (first) face in the cluster
                val similarity = faceHelper.compareFaces(
                    face.embedding,
                    cluster[0].embedding,
                )

                if (similarity > bestSimilarity) {
                    bestSimilarity = similarity
                    bestClusterIdx = clusterIdx
                }
            }

            if (bestClusterIdx >= 0) {
                clusters[bestClusterIdx].add(face)
            } else {
                clusters.add(mutableListOf(face))
            }
        }

        // Convert to FaceGroupResult, sorted by size
        return clusters
            .mapIndexed { index, cluster ->
                FaceGroupResult(
                    groupId = index,
                    faces = cluster,
                    representativeFace = cluster[0],
                )
            }
            .sortedByDescending { it.faces.size }
    }

    /**
     * Load a bitmap from a URI.
     */
    private fun loadBitmap(uri: Uri): Bitmap? {
        return try {
            val inputStream = context.contentResolver.openInputStream(uri)
            BitmapFactory.decodeStream(inputStream).also {
                inputStream?.close()
            }
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Release all resources.
     */
    fun close() {
        faceHelper.close()
        faceDetector.close()
    }
}
