# To use GPU with TensorFlow, download and install CUDA and cuDNN:
# Recommended for TensorFlow 2.18.1: CUDA 12.3 and cuDNN 8.9 (see https://www.tensorflow.org/install/source#gpu)
# 1. Visit https://developer.nvidia.com/cuda-downloads and select your OS to download CUDA Toolkit 12.3.
# 2. Install CUDA Toolkit following NVIDIA's instructions.
# 3. Download cuDNN 8.9 from https://developer.nvidia.com/cudnn (requires free NVIDIA Developer account).
# 4. Extract cuDNN files and copy them into your CUDA installation directories.
# 5. Ensure your GPU drivers are up to date.
# 6. Verify installation with: `tf.config.list_physical_devices('GPU')`
# For more details, see: https://www.tensorflow.org/install/gpu

import tensorflow as tf
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import CONSTANTS as CONST

data_dir = "C:\\Games\\site\\prectic\\ExtructedFrames extended"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs detected: {[gpu.name for gpu in gpus]}")
        tf.debugging.set_log_device_placement(True)
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected by TensorFlow. Training will use CPU.")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=CONST.IMG_SIZE,  # Ensure correct input size
    batch_size=CONST.BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=CONST.IMG_SIZE,  # Fix: add target_size for validation
    batch_size=CONST.BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print("Класи:", train_generator.class_indices)

base_model = MobileNet(weights='imagenet', include_top=False, input_shape=CONST.IMG_SIZE + (3,))

# Freeze all layers initially
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)  # Reduced size for efficiency
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

class TerminateOnValAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, target=0.90):
        super().__init__()
        self.target = target

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy')
        if val_acc is not None and val_acc >= self.target:
            print(f"\nReached val_accuracy {val_acc:.4f} at epoch {epoch+1}. Stopping training.")
            self.model.stop_training = True

# Train with early termination callback
history = model.fit(
    train_generator,
    epochs=CONST.EPOCHS,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[TerminateOnValAccuracy(target=0.90)]
)

history_df = pd.DataFrame(history.history)
history_df.to_csv("training_history.csv", index=False)
print("Training history saved to 'training_history.csv'.")

model.save("classification_model.keras")
print("Модель збережена у файл 'classification_model.keras'")


