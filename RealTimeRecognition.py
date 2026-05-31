import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
import matplotlib.pyplot as plt

# =========================================================
# SETTINGS
# =========================================================

HAGRID_DATASET_PATH = r"D:\hagrid-sample-500k-384p\hagrid-sample-500k-384p\hagrid_500k"

IMG_SIZE = 224
MAX_IMAGES_PER_CLASS = 6000
BATCH_SIZE = 32  # Keeping your efficient CPU step-overhead configuration
INITIAL_EPOCHS = 12
FINE_TUNE_EPOCHS = 20

# =========================================================
# PREPARE FILE PATHS (RAM-FRIENDLY)
# =========================================================

class_names = sorted([
    f for f in os.listdir(HAGRID_DATASET_PATH)
    if os.path.isdir(os.path.join(HAGRID_DATASET_PATH, f))
])

print(f"\nFound {len(class_names)} classes:")
print(class_names)

file_paths = []
labels = []

for class_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(HAGRID_DATASET_PATH, class_name)
    file_list = [
        os.path.join(class_dir, f) for f in os.listdir(class_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ][:MAX_IMAGES_PER_CLASS]
    
    file_paths.extend(file_list)
    labels.extend([class_idx] * len(file_list))

file_paths = np.array(file_paths)
labels = np.array(labels, dtype=np.int32)

# Shuffle paths cleanly before splitting
indices = np.arange(len(file_paths))
np.random.seed(42)
np.random.shuffle(indices)
file_paths = file_paths[indices]
labels = labels[indices]

# Train / Val Split (80% / 20%)
split_idx = int(0.8 * len(file_paths))
train_paths, val_paths = file_paths[:split_idx], file_paths[split_idx:]
train_labels, val_labels = labels[:split_idx], labels[split_idx:]

# =========================================================
# CPU-SAFE DATA PIPELINE (TENSORFLOW DATASET)
# =========================================================

def load_and_preprocess_image(path, label):
    """ Reads images lazily from disk to prevent CPU/RAM overheating """
    img_raw = tf.io.read_file(path)
    img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
    img_resized = tf.image.resize(img_tensor, [IMG_SIZE, IMG_SIZE])
    # FIX: MobileNetV3 expects [0, 255] float/int pixels, explicitly cast to float32 without division
    return tf.cast(img_resized, tf.float32), label

# Create lightweight streaming pipelines
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# =========================================================
# VISUALIZE DATA (RESTORED FROM CODE 1)
# =========================================================
# Take a small batch to preview dataset structural images safely
preview_samples = list(train_ds.take(10).as_numpy_iterator())

plt.figure(figsize=(12, 6))
for i, (img, lbl) in enumerate(preview_samples):
    plt.subplot(2, 5, i + 1)
    plt.imshow(img.astype(np.uint8))  # Cast back to int for clear plotting
    plt.title(class_names[lbl], fontsize=9)
    plt.axis("off")
plt.suptitle("Training Samples Pipeline Preview")
plt.tight_layout()
plt.show()

# =========================================================
# DATA AUGMENTATION (RUNS ON DATASET)
# =========================================================

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
    layers.RandomContrast(0.10)
])

# Batch, augment, and prefetch to keep CPU cool
train_ds = train_ds.shuffle(buffer_size=1024).batch(BATCH_SIZE)
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

val_ds = val_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# =========================================================
# UPGRADED BASE MODEL (MOBILENETV3 LARGE)
# =========================================================

base_model = tf.keras.applications.MobileNetV3Large(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# =========================================================
# BUILD NETWORK ARCHITECTURE
# =========================================================

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.BatchNormalization()(x) 
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = models.Model(inputs, outputs)

# =========================================================
# COMPILE & CALLBACKS
# =========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4), # Refined to prevent initial convergence overshooting
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stopping = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6)
checkpoint = ModelCheckpoint("checkpoint_hagrid.keras", monitor="val_accuracy", save_best_only=True, verbose=1)

callbacks = [early_stopping, reduce_lr, checkpoint]

# =========================================================
# INITIAL TRAINING
# =========================================================

print("\nStarting cool-running initial training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks,
    shuffle=True
)

# =========================================================
# FINE-TUNING (UNFREEZE STRATEGIC TOP LAYERS ONLY)
# =========================================================

print("\nStarting fine-tuning...")
base_model.trainable = True

# Freeze earlier blocks to lock structural weights, leaving the last 60 for gesture nuance adjustments
for layer in base_model.layers[:-60]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks,
    shuffle=True
)

# =========================================================
# SAVE MODEL
# =========================================================

model.save("new_hagrid_model.keras")

with open("hagrid_classes.pkl", "wb") as f:
    pickle.dump(class_names, f)

print("\nTraining completely finished! Model saved safely.")

# =========================================================
# MERGE HISTORIES & PLOTS (RESTORED FROM CODE 1)
# =========================================================

acc = history.history["accuracy"] + history_fine.history["accuracy"]
val_acc = history.history["val_accuracy"] + history_fine.history["val_accuracy"]

loss = history.history["loss"] + history_fine.history["loss"]
val_loss = history.history["val_loss"] + history_fine.history["val_loss"]

epochs_range = range(len(acc))

plt.figure(figsize=(14, 6))

# Accuracy Subplot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy", linewidth=2)
plt.plot(epochs_range, val_acc, label="Validation Accuracy", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(alpha=0.3)

# Loss Subplot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss", linewidth=2)
plt.plot(epochs_range, val_loss, label="Validation Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

