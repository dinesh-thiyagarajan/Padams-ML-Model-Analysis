package com.example.facerecognition

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.net.Uri
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Helper class for face recognition using TFLite on Android.
 *
 * This class handles:
 * 1. Loading the TFLite face embedding model
 * 2. Preprocessing face images
 * 3. Extracting 128-d face embeddings
 * 4. Comparing faces using cosine similarity
 *
 * Usage:
 * ```kotlin
 * val helper = FaceRecognitionHelper(context)
 * val embedding = helper.getEmbedding(faceBitmap)
 * val similarity = helper.compareFaces(embedding1, embedding2)
 * ```
 */
class FaceRecognitionHelper(
    private val context: Context,
    private val modelFileName: String = "face_embedder.tflite",
    private val inputSize: Int = 112,
    private val embeddingSize: Int = 128,
) {
    private var interpreter: Interpreter? = null

    init {
        loadModel()
    }

    /**
     * Load the TFLite model from assets.
     */
    private fun loadModel() {
        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }
        interpreter = Interpreter(loadModelFile(), options)
    }

    /**
     * Load the TFLite model file from assets as a MappedByteBuffer.
     */
    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelFileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Preprocess a face bitmap for the embedding model.
     *
     * Steps:
     * 1. Resize to 112x112
     * 2. Convert to float32
     * 3. Normalize pixel values to [-1, 1]
     *
     * @param bitmap Face crop as a Bitmap (any size)
     * @return ByteBuffer ready for TFLite inference
     */
    private fun preprocessFace(bitmap: Bitmap): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in pixels) {
            // Extract RGB and normalize to [-1, 1]
            val r = ((pixel shr 16) and 0xFF).toFloat()
            val g = ((pixel shr 8) and 0xFF).toFloat()
            val b = (pixel and 0xFF).toFloat()

            byteBuffer.putFloat((r - 127.5f) / 128.0f)
            byteBuffer.putFloat((g - 127.5f) / 128.0f)
            byteBuffer.putFloat((b - 127.5f) / 128.0f)
        }

        if (resized != bitmap) {
            resized.recycle()
        }

        return byteBuffer
    }

    /**
     * Extract a 128-d face embedding from a face bitmap.
     *
     * @param faceBitmap Cropped face image
     * @return FloatArray of size 128 (L2-normalized embedding)
     */
    fun getEmbedding(faceBitmap: Bitmap): FloatArray {
        val input = preprocessFace(faceBitmap)
        val output = Array(1) { FloatArray(embeddingSize) }

        interpreter?.run(input, output)

        return output[0]
    }

    /**
     * Compute cosine similarity between two face embeddings.
     *
     * @return Similarity score in [-1, 1]. Higher = more similar.
     *         Typically > 0.5 means same person.
     */
    fun compareFaces(embedding1: FloatArray, embedding2: FloatArray): Float {
        var dotProduct = 0f
        var norm1 = 0f
        var norm2 = 0f

        for (i in embedding1.indices) {
            dotProduct += embedding1[i] * embedding2[i]
            norm1 += embedding1[i] * embedding1[i]
            norm2 += embedding2[i] * embedding2[i]
        }

        val denominator = sqrt(norm1) * sqrt(norm2)
        return if (denominator > 0) dotProduct / denominator else 0f
    }

    /**
     * Check if two face embeddings belong to the same person.
     *
     * @param embedding1 First face embedding
     * @param embedding2 Second face embedding
     * @param threshold Similarity threshold (default 0.5)
     * @return true if faces likely belong to the same person
     */
    fun isSamePerson(
        embedding1: FloatArray,
        embedding2: FloatArray,
        threshold: Float = 0.5f,
    ): Boolean {
        return compareFaces(embedding1, embedding2) > threshold
    }

    /**
     * Crop a face from an image given a bounding box.
     *
     * @param image Source image bitmap
     * @param faceRect Bounding box of the detected face
     * @param margin Fractional margin to add around the face (0.2 = 20%)
     * @return Cropped face bitmap
     */
    fun cropFace(image: Bitmap, faceRect: Rect, margin: Float = 0.2f): Bitmap {
        val faceWidth = faceRect.width()
        val faceHeight = faceRect.height()

        val marginX = (faceWidth * margin).toInt()
        val marginY = (faceHeight * margin).toInt()

        val left = max(0, faceRect.left - marginX)
        val top = max(0, faceRect.top - marginY)
        val right = min(image.width, faceRect.right + marginX)
        val bottom = min(image.height, faceRect.bottom + marginY)

        return Bitmap.createBitmap(image, left, top, right - left, bottom - top)
    }

    /**
     * Release resources.
     */
    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
