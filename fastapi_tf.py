from __future__ import print_function
from PIL import Image
import tensorflow as tf
import numpy as np
from io import BytesIO
from tensorflow import keras
import uvicorn
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

model = None

# web: uvicorn fastapi_tf:app


def load_model():
    model = tf.keras.models.load_model('gender.h5')
    print("Model loaded")
    return model


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    image.save('file.png')
    return image


def predict2(image: Image.Image):
    sunflower_path = 'file.png'

    class_names = ['female', 'male']
    img_height = 398
    img_width = 309

    img = keras.preprocessing.image.load_img(
        sunflower_path, target_size=(img_height, img_width)
    )

    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    model = tf.keras.models.load_model('gender.h5')

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

    return [class_names[np.argmax(score)], 100 * np.max(score)]


@ app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    print(file)
    print(type(file))
    image = read_imagefile(await file.read())
    prediction = predict2(image)

    return prediction

if __name__ == "__main__":
    uvicorn.run(app, host="192.168.1.8", debug=True)
