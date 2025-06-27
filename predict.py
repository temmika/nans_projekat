import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class_labels = ['banana', 'carambola', 'mango', 'peach', 'pitaya']

model = load_model('fruit_model.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    print(f"Predikcija: {predicted_class}")

# Primer:
#predict_image("data/test/Banana062.png")
