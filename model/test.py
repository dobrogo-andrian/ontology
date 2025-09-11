from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import CONSTANTS as CONST


model = load_model(r"../model/classification_model.keras")

image_path = r"C:\Games\site\prectic\ExtructedFrames extended\BMP\20240920233644\frame_00001.jpg"
image = load_img(image_path, target_size=CONST.IMG_SIZE)
image_array = img_to_array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)

predictions = model.predict(image_array)
predicted_class = np.argmax(predictions)

class_labels = ['BMP', 'Bradley', 'Tank', 'unknown']

print(f"Прогнозована категорія: {class_labels[predicted_class]}")