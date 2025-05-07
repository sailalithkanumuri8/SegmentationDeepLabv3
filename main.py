import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tqdm import tqdm

tf.random.set_seed(42)
np.random.seed(42)


# =======================================
# 1. Define DeepLabV3+ Model Architecture
# =======================================

def DilatedSpatialPyramidPooling(inputs):
    """Atrous Spatial Pyramid Pooling with different dilation rates"""
    dims = inputs.shape

    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(inputs)
    x = layers.Conv2D(256, 1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    # 1x1 convolution branch
    branch_1 = layers.Conv2D(256, 1, padding="same")(inputs)
    branch_1 = layers.BatchNormalization()(branch_1)
    branch_1 = layers.Activation("relu")(branch_1)

    # 3x3 convolution with dilation rate 6
    branch_2 = layers.Conv2D(256, 3, padding="same", dilation_rate=6)(inputs)
    branch_2 = layers.BatchNormalization()(branch_2)
    branch_2 = layers.Activation("relu")(branch_2)

    # 3x3 convolution with dilation rate 12
    branch_3 = layers.Conv2D(256, 3, padding="same", dilation_rate=12)(inputs)
    branch_3 = layers.BatchNormalization()(branch_3)
    branch_3 = layers.Activation("relu")(branch_3)

    # 3x3 convolution with dilation rate 18
    branch_4 = layers.Conv2D(256, 3, padding="same", dilation_rate=18)(inputs)
    branch_4 = layers.BatchNormalization()(branch_4)
    branch_4 = layers.Activation("relu")(branch_4)

    # Concatenate all branches
    x = layers.Concatenate()([x, branch_1, branch_2, branch_3, branch_4])
    x = layers.Conv2D(256, 1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def DeepLabV3Plus(image_size, num_classes):
    """DeepLabV3+ model with ResNet50 backbone"""
    model_input = keras.Input(shape=(image_size, image_size, 3))

    # Use ResNet50 as backbone
    preprocessed = keras.applications.resnet50.preprocess_input(model_input)
    resnet50 = keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_tensor=preprocessed
    )

    # Freeze early layers for transfer learning
    for layer in resnet50.layers[:100]:
        layer.trainable = False

    # Add L2 regularization to all trainable layers
    regularizer = tf.keras.regularizers.l2(0.0001)

    # Get features from ResNet50
    x = resnet50.get_layer("conv4_block6_2_relu").output

    # Apply ASPP with regularization
    x = DilatedSpatialPyramidPooling(x)

    # Upsampling path
    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)

    # Get low-level features from ResNet50
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = layers.Conv2D(48, 1, padding="same", kernel_regularizer=regularizer)(input_b)
    input_b = layers.BatchNormalization()(input_b)
    input_b = layers.Activation("relu")(input_b)

    # Concatenate upsampled ASPP features with low-level features
    x = layers.Concatenate()([input_a, input_b])

    # Final convolutions with regularization
    x = layers.Conv2D(256, 3, padding="same", kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)  # Add dropout

    x = layers.Conv2D(256, 3, padding="same", kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)  # Add dropout

    # Final upsampling to original image size
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)

    # Output layer
    if num_classes == 1:
        # Binary segmentation
        x = layers.Conv2D(num_classes, 1, padding="same", kernel_regularizer=regularizer)(x)
        x = layers.Activation("sigmoid")(x)
    else:
        # Multi-class segmentation
        x = layers.Conv2D(num_classes, 1, padding="same", kernel_regularizer=regularizer)(x)
        x = layers.Activation("softmax")(x)

    return keras.Model(inputs=model_input, outputs=x)


# =======================================
# 2. Data Preparation Functions
# =======================================

def load_dataset(image_dir, mask_dir, image_size, num_classes, limit=None):
    """
    Load images and masks from directories

    Args:
        image_dir: Directory containing images
        mask_dir: Directory containing corresponding masks
        image_size: Size to resize images to
        num_classes: Number of segmentation classes
        limit: Optional limit on number of images to load

    Returns:
        images: Array of images
        masks: Array of masks
    """
    image_files = sorted(
        [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted(
        [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if limit:
        image_files = image_files[:limit]
        mask_files = mask_files[:limit]

    images = []
    masks = []

    for img_path, mask_path in tqdm(zip(image_files, mask_files), total=len(image_files), desc="Loading dataset"):
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size))
        img = img.astype(np.float32) / 255.0

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

        # For binary segmentation
        if num_classes == 1:
            mask = (mask > 0).astype(np.float32)
        else:
            # For multi-class segmentation, ensure mask values are 0 to num_classes-1
            mask = np.clip(mask, 0, num_classes - 1)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)


def create_data_generators(images, masks, batch_size, num_classes, validation_split=0.2):
    """
    Create training and validation data generators with data augmentation

    Args:
        images: Array of images
        masks: Array of masks
        batch_size: Batch size for training
        num_classes: Number of segmentation classes
        validation_split: Fraction of data to use for validation

    Returns:
        train_gen: Training data generator
        val_gen: Validation data generator
    """
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, masks, test_size=validation_split, random_state=42
    )

    # Data augmentation for training
    data_gen_args = dict(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Create image data generator
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Create generators
    seed = 42
    image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=seed)

    # Combine generators
    train_generator = zip(image_generator, mask_generator)

    # No augmentation for validation data
    val_datagen = ImageDataGenerator()
    val_image_generator = val_datagen.flow(X_val, batch_size=batch_size, seed=seed)
    val_mask_generator = val_datagen.flow(y_val, batch_size=batch_size, seed=seed)
    val_generator = zip(val_image_generator, val_mask_generator)

    return train_generator, val_generator, len(X_train), len(X_val)


# Custom data generator for TensorFlow Dataset API
def create_tf_dataset(images, masks, batch_size, num_classes, is_training=True):
    """Create TensorFlow Dataset for training/validation"""

    # Convert masks to the right format based on num_classes
    if num_classes > 1:
        # For multi-class segmentation, use one-hot encoding
        masks_processed = to_categorical(masks, num_classes=num_classes)
    else:
        # For binary segmentation, expand dimensions
        masks_processed = np.expand_dims(masks, axis=-1)

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, masks_processed))

    # Apply augmentation if training
    if is_training:
        def augment(image, mask):
            # Random flip
            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_left_right(image)
                mask = tf.image.flip_left_right(mask)

            # Random flip up-down
            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_up_down(image)
                mask = tf.image.flip_up_down(mask)

            # Random brightness
            image = tf.image.random_brightness(image, max_delta=0.2)

            # Random contrast
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

            # Random saturation
            image = tf.image.random_saturation(image, lower=0.8, upper=1.2)

            # Random hue
            image = tf.image.random_hue(image, max_delta=0.1)

            # Ensure image values are in [0, 1]
            image = tf.clip_by_value(image, 0, 1)

            return image, mask

        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


# =======================================
# 3. Training and Evaluation Functions
# =======================================

def train_model(model, train_dataset, val_dataset, epochs, steps_per_epoch, validation_steps):
    """Train the DeepLabV3+ model"""

    # Define learning rate schedule with warmup
    initial_learning_rate = 0.001
    warmup_epochs = 5
    decay_epochs = epochs - warmup_epochs

    def warmup_cosine_decay_schedule(epoch):
        # Warmup phase
        if epoch < warmup_epochs:
            return initial_learning_rate * ((epoch + 1) / warmup_epochs)
        # Cosine decay phase
        else:
            progress = (epoch - warmup_epochs) / decay_epochs
            cosine_decay = 0.5 * (1 + tf.cos(3.14159 * progress))
            return initial_learning_rate * cosine_decay

    # Define callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "deeplabv3plus_model.h5",
            save_best_only=True,
            monitor="val_loss"
        ),
        keras.callbacks.LearningRateScheduler(warmup_cosine_decay_schedule),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir="logs/deeplabv3plus"
        )
    ]

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    return history, model


def evaluate_model(model, test_images, test_masks, num_classes):
    """Evaluate the model and visualize results"""

    # Make predictions
    predictions = model.predict(test_images)

    # Visualize results
    fig, axes = plt.subplots(len(test_images), 3, figsize=(15, 5 * len(test_images)))

    for i in range(len(test_images)):
        # Display original image
        axes[i, 0].imshow(test_images[i])
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")

        # Display ground truth mask
        if num_classes == 1:
            axes[i, 1].imshow(test_masks[i], cmap="gray")
        else:
            axes[i, 1].imshow(test_masks[i], cmap="viridis")
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

        # Display predicted mask
        if num_classes == 1:
            pred_mask = predictions[i] > 0.5
            axes[i, 2].imshow(pred_mask.squeeze(), cmap="gray")
        else:
            pred_mask = np.argmax(predictions[i], axis=-1)
            axes[i, 2].imshow(pred_mask, cmap="viridis")
        axes[i, 2].set_title("Predicted Mask")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("segmentation_results.png")
    plt.show()

    return predictions


# =======================================
# 4. Main Execution
# =======================================

def main():
    # Configuration
    IMAGE_SIZE = 256
    BATCH_SIZE = 4
    EPOCHS = 100
    NUM_CLASSES = 21
    INITIAL_LEARNING_RATE = 0.001  # Reduced from 0.01

    # Enable memory growth for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Paths to your dataset
    IMAGE_DIR = "path/to/images"
    MASK_DIR = "path/to/masks"

    # Check if we're using sample data
    use_sample_data = True

    if use_sample_data:
        print("Using sample data (Pascal VOC subset)...")
        # Create synthetic data for demonstration
        num_samples = 100
        images = np.random.rand(num_samples, IMAGE_SIZE, IMAGE_SIZE, 3)

        if NUM_CLASSES == 1:
            # Binary masks
            masks = np.random.randint(0, 2, size=(num_samples, IMAGE_SIZE, IMAGE_SIZE))
        else:
            # Multi-class masks with more structured patterns
            masks = np.zeros((num_samples, IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32)
            for i in range(num_samples):
                # Create some structured patterns
                center_x = np.random.randint(0, IMAGE_SIZE)
                center_y = np.random.randint(0, IMAGE_SIZE)
                radius = np.random.randint(20, 50)
                y, x = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
                dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                mask = dist_from_center <= radius
                masks[i] = mask.astype(np.int32) * np.random.randint(1, NUM_CLASSES)

        print(f"Created {num_samples} synthetic samples for demonstration")
    else:
        # Load real dataset
        print("Loading dataset from disk...")
        images, masks = load_dataset(IMAGE_DIR, MASK_DIR, IMAGE_SIZE, NUM_CLASSES)
        print(f"Loaded {len(images)} images and masks")

    # Split data
    train_idx, val_idx = train_test_split(
        range(len(images)), test_size=0.2, random_state=42
    )

    train_images, train_masks = images[train_idx], masks[train_idx]
    val_images, val_masks = images[val_idx], masks[val_idx]

    # Create TensorFlow datasets
    train_dataset = create_tf_dataset(train_images, train_masks, BATCH_SIZE, NUM_CLASSES, is_training=True)
    val_dataset = create_tf_dataset(val_images, val_masks, BATCH_SIZE, NUM_CLASSES, is_training=False)

    # Calculate steps per epoch
    steps_per_epoch = len(train_images) // BATCH_SIZE
    validation_steps = len(val_images) // BATCH_SIZE

    # Create model
    print("Creating DeepLabV3+ model...")
    model = DeepLabV3Plus(IMAGE_SIZE, NUM_CLASSES)

    # Compile model with mixed precision
    if NUM_CLASSES == 1:
        # Binary segmentation
        loss = keras.losses.BinaryCrossentropy()
        metrics = [
            tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name="iou"),
            "accuracy"
        ]
    else:
        # Multi-class segmentation
        loss = keras.losses.CategoricalCrossentropy()
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES, name="mean_iou")
        ]

    # Enable mixed precision training
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Create optimizer with learning rate schedule and gradient clipping
    optimizer = keras.optimizers.Adam(
        learning_rate=INITIAL_LEARNING_RATE,
        clipnorm=1.0  # Add gradient clipping
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    # Print model summary
    model.summary()

    # Train model
    print("Training model...")
    history, trained_model = train_model(
        model,
        train_dataset,
        val_dataset,
        EPOCHS,
        steps_per_epoch,
        validation_steps
    )

    # Save the model
    trained_model.save("deeplabv3plus_final_model.h5")
    print("Model saved as 'deeplabv3plus_final_model.h5'")

    # Evaluate on a few test samples
    print("Evaluating model...")
    test_samples = 3
    test_indices = np.random.choice(len(val_images), test_samples, replace=False)
    test_images = val_images[test_indices]
    test_masks = val_masks[test_indices]

    predictions = evaluate_model(trained_model, test_images, test_masks, NUM_CLASSES)

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()


# =======================================
# 5. Inference Function
# =======================================

def predict_segmentation(model_path, image_path, image_size, num_classes):
    """
    Perform segmentation on a new image using a trained model

    Args:
        model_path: Path to the saved model
        image_path: Path to the image to segment
        image_size: Size to resize the image to
        num_classes: Number of segmentation classes

    Returns:
        Original image and segmentation mask
    """
    # Load model
    model = keras.models.load_model(model_path)

    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (image_size, image_size))
    img_normalized = img_resized.astype(np.float32) / 255.0

    # Make prediction
    prediction = model.predict(np.expand_dims(img_normalized, axis=0))[0]

    # Process prediction based on number of classes
    if num_classes == 1:
        # Binary segmentation
        mask = (prediction > 0.5).astype(np.uint8)
        colored_mask = np.zeros_like(img_resized)
        colored_mask[mask.squeeze() > 0] = [255, 0, 0]  # Red for foreground
    else:
        # Multi-class segmentation
        mask = np.argmax(prediction, axis=-1)

        # Create a colored mask
        colored_mask = np.zeros_like(img_resized)

        # Assign different colors to different classes
        for class_idx in range(1, num_classes):  # Skip background (0)
            # Generate a random color for each class
            color = np.array([
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            ])
            colored_mask[mask == class_idx] = color

    # Overlay mask on original image
    alpha = 0.6
    overlay = cv2.addWeighted(img_resized, 1 - alpha, colored_mask, alpha, 0)

    # Display results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_resized)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    if num_classes == 1:
        plt.imshow(mask.squeeze(), cmap="gray")
    else:
        plt.imshow(mask, cmap="viridis")
    plt.title("Segmentation Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("inference_result.png")
    plt.show()

    return img_resized, mask, overlay


# =======================================
# 6. Example Usage
# =======================================

if __name__ == "__main__":
    # Train the model
    main()

    """
    # For inference on a new image
    IMAGE_SIZE = 512
    NUM_CLASSES = 21  # Adjust based on your model

    # Replace with your model path and test image
    predict_segmentation(
        model_path="deeplabv3plus_final_model.h5",
        image_path="path/to/test/image.jpg",
        image_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES
    )
    """