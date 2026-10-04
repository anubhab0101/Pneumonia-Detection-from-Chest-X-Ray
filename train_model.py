import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np

# Set image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data directories
base_dir = 'chest_xray'

# Create data generators
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)
val_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

try:
    train_generator = train_datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        os.path.join(base_dir, 'val'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    # Calculate class weights for imbalance
    counts = np.bincount(train_generator.classes)
    weight_for_0 = (1 / counts[0]) * (len(train_generator.classes) / 2.0)
    weight_for_1 = (1 / counts[1]) * (len(train_generator.classes) / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}

except Exception as e:
    print(f"Dataset not found or error occurred: {e}")
    train_generator, val_generator, class_weight = None, None, None


# Build Transfer Learning Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

# Train the model
if train_generator and val_generator:
    epochs = 15
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks
    )

    # Save the final best model
    model.save('best_model.h5')
    print("Model saved as 'best_model.h5'")
else:
    print("Skipping training as dataset is missing.")
