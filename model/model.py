import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import CONSTANTS as CONST

data_dir = "C:\\Games\\site\\prectic\\ExtructedFrames extended"

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
    target_size=CONST.IMG_SIZE,
    batch_size=CONST.BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=CONST.IMG_SIZE,
    batch_size=CONST.BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print("Класи:", train_generator.class_indices)

base_model = MobileNet(weights='imagenet', include_top=False, input_shape=CONST.IMG_SIZE + (3,))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
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
            print(f"\nReached val_accuracy {val_acc:.4f} at epoch {epoch + 1}. Stopping training.")
            self.model.stop_training = True


history = model.fit(
    train_generator,
    epochs=CONST.EPOCHS,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[TerminateOnValAccuracy(target=0.90)]
)

history_df = pd.DataFrame(history.history)
history_df.to_csv("training_history.csv", index=False)
print("Історія навчання збережена в 'training_history.csv'.")

model.save("classification_model.keras")
print("Модель збережена в 'classification_model.keras'")
