"""Face embedding extraction using MobileFaceNet architecture.

MobileFaceNet is a lightweight face recognition model optimized for mobile
devices. It uses depthwise separable convolutions and produces compact 128-d
embedding vectors that can be compared using cosine similarity or L2 distance.
"""

from typing import List, Optional

import numpy as np
import tensorflow as tf

from face_recognition.detector import DetectedFace
from face_recognition.utils import preprocess_face


def _depthwise_separable_conv(
    x: tf.Tensor,
    filters: int,
    stride: int = 1,
    name: str = "",
) -> tf.Tensor:
    """Depthwise separable convolution block."""
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_dw",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2], name=f"{name}_dw_prelu")(x)

    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        name=f"{name}_pw",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_pw_bn")(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2], name=f"{name}_pw_prelu")(x)

    return x


def _bottleneck_residual(
    x: tf.Tensor,
    filters: int,
    stride: int = 1,
    expansion: int = 1,
    name: str = "",
) -> tf.Tensor:
    """Inverted residual bottleneck block."""
    in_channels = x.shape[-1]
    expanded = expansion * in_channels

    # Expansion
    shortcut = x
    x = tf.keras.layers.Conv2D(
        expanded,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        name=f"{name}_expand",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_expand_bn")(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2], name=f"{name}_expand_prelu")(x)

    # Depthwise
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_depthwise",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_depthwise_bn")(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2], name=f"{name}_depthwise_prelu")(x)

    # Projection
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        name=f"{name}_project",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_project_bn")(x)

    # Residual connection (only when dimensions match)
    if stride == 1 and in_channels == filters:
        x = tf.keras.layers.Add(name=f"{name}_add")([shortcut, x])

    return x


def build_mobilefacenet(
    input_shape: tuple = (112, 112, 3),
    embedding_size: int = 128,
) -> tf.keras.Model:
    """Build the MobileFaceNet model architecture.

    Architecture follows the MobileFaceNet paper with modifications:
    - Input: 112x112x3 face image
    - Output: 128-d L2-normalized embedding vector

    Args:
        input_shape: Input image shape (H, W, C).
        embedding_size: Size of the output embedding vector.

    Returns:
        A Keras model that maps face images to embedding vectors.
    """
    inputs = tf.keras.Input(shape=input_shape, name="face_input")

    # Initial convolution: 112x112 -> 56x56
    x = tf.keras.layers.Conv2D(
        64, kernel_size=3, strides=2, padding="same", use_bias=False, name="conv1"
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="conv1_bn")(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2], name="conv1_prelu")(x)

    # Depthwise separable: 56x56
    x = _depthwise_separable_conv(x, 64, stride=1, name="dw_conv1")

    # Bottleneck blocks
    # Stage 1: 56x56 -> 28x28, 64 channels
    for i in range(5):
        stride = 2 if i == 0 else 1
        x = _bottleneck_residual(x, 64, stride=stride, expansion=2, name=f"stage1_{i}")

    # Stage 2: 28x28 -> 14x14, 128 channels
    for i in range(1):
        x = _bottleneck_residual(x, 128, stride=2, expansion=4, name=f"stage2_{i}")

    # Stage 3: 14x14 -> 7x7, 128 channels
    for i in range(6):
        stride = 2 if i == 0 else 1
        x = _bottleneck_residual(x, 128, stride=stride, expansion=2, name=f"stage3_{i}")

    # Stage 4: 7x7, 128 channels
    for i in range(2):
        x = _bottleneck_residual(x, 128, stride=1, expansion=4, name=f"stage4_{i}")

    # Final conv: 1x1 to 512
    x = tf.keras.layers.Conv2D(
        512, kernel_size=1, strides=1, padding="valid", use_bias=False, name="conv2"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="conv2_bn")(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2], name="conv2_prelu")(x)

    # Global depthwise conv (GDConv) - replaces global average pooling
    # This is a key feature of MobileFaceNet for better performance
    kernel_size = x.shape[1]  # Should be 7
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=1,
        padding="valid",
        use_bias=False,
        name="gdconv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="gdconv_bn")(x)

    # Flatten and produce embedding
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(embedding_size, use_bias=False, name="embedding_dense")(x)
    x = tf.keras.layers.BatchNormalization(name="embedding_bn")(x)

    # L2 normalize the embedding
    embeddings = tf.keras.layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=1),
        name="l2_normalize",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=embeddings, name="MobileFaceNet")
    return model


class FaceEmbedder:
    """Extracts face embeddings using MobileFaceNet.

    The embedder takes cropped face images and produces compact 128-d vectors.
    Faces of the same person will have similar embeddings (high cosine similarity),
    while faces of different people will have dissimilar embeddings.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_size: int = 128,
        input_size: tuple = (112, 112),
    ):
        self.embedding_size = embedding_size
        self.input_size = input_size
        self._model = None
        self._model_path = model_path
        self._tflite_interpreter = None

    def _load_model(self):
        """Load or build the embedding model."""
        if self._model is not None:
            return

        if self._model_path and self._model_path.endswith(".tflite"):
            self._load_tflite_model(self._model_path)
        elif self._model_path and tf.io.gfile.exists(self._model_path):
            self._model = tf.keras.models.load_model(self._model_path)
        else:
            # Build a fresh model (pretrained weights can be loaded later)
            self._model = build_mobilefacenet(
                input_shape=(*self.input_size, 3),
                embedding_size=self.embedding_size,
            )

    def _load_tflite_model(self, tflite_path: str):
        """Load a TFLite model for inference."""
        self._tflite_interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self._tflite_interpreter.allocate_tensors()

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract embedding from a single face image.

        Args:
            face_image: RGB face crop as numpy array.

        Returns:
            128-d embedding vector as numpy array.
        """
        self._load_model()

        preprocessed = preprocess_face(face_image, target_size=self.input_size)
        batch = np.expand_dims(preprocessed, axis=0)

        if self._tflite_interpreter is not None:
            return self._run_tflite(batch)

        embedding = self._model.predict(batch, verbose=0)
        return embedding[0]

    def _run_tflite(self, batch: np.ndarray) -> np.ndarray:
        """Run inference with the TFLite interpreter."""
        input_details = self._tflite_interpreter.get_input_details()
        output_details = self._tflite_interpreter.get_output_details()

        self._tflite_interpreter.set_tensor(
            input_details[0]["index"],
            batch.astype(np.float32),
        )
        self._tflite_interpreter.invoke()

        embedding = self._tflite_interpreter.get_tensor(output_details[0]["index"])
        return embedding[0]

    def get_embeddings_batch(
        self,
        face_images: List[np.ndarray],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Extract embeddings from multiple face images.

        Args:
            face_images: List of RGB face crops.
            batch_size: Batch size for inference.

        Returns:
            Array of shape (N, embedding_size) with all embeddings.
        """
        self._load_model()

        all_embeddings = []

        for i in range(0, len(face_images), batch_size):
            batch_images = face_images[i : i + batch_size]
            preprocessed = np.array(
                [preprocess_face(img, target_size=self.input_size) for img in batch_images]
            )

            if self._tflite_interpreter is not None:
                # TFLite processes one at a time
                embeddings = np.array(
                    [self._run_tflite(preprocessed[j : j + 1]) for j in range(len(preprocessed))]
                )
            else:
                embeddings = self._model.predict(preprocessed, verbose=0)

            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings, axis=0)

    def get_embeddings_from_detections(
        self,
        faces: List[DetectedFace],
    ) -> np.ndarray:
        """Extract embeddings from a list of DetectedFace objects.

        Args:
            faces: List of DetectedFace objects with face_crop set.

        Returns:
            Array of shape (N, embedding_size) with embeddings.
        """
        face_images = []
        for face in faces:
            if face.face_crop is not None:
                face_images.append(face.face_crop)

        if not face_images:
            return np.array([]).reshape(0, self.embedding_size)

        return self.get_embeddings_batch(face_images)

    def save_model(self, path: str):
        """Save the Keras model to disk."""
        self._load_model()
        if self._model is not None:
            self._model.save(path)

    @property
    def model(self) -> tf.keras.Model:
        """Access the underlying Keras model."""
        self._load_model()
        return self._model
