import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import CONSTANTS as CONST
from sklearn.metrics import classification_report

model = load_model(r"../model/classification_model.keras")

data_dir = r"C:\Games\site\prectic\ExtructedFrames extended"
val_datagen = ImageDataGenerator(rescale=1.0/255)
validation_generator = val_datagen.flow_from_directory(
    data_dir,
    target_size=CONST.IMG_SIZE,
    batch_size=CONST.BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    subset=None
)

predictions = model.predict(validation_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = validation_generator.classes

class_labels = list(validation_generator.class_indices.keys())


print(classification_report(y_true, y_pred, target_names=class_labels))