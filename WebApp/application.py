import os
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import cv2
from flask import Flask, request, jsonify, render_template

# Create flask application
application = Flask(__name__)

dict = {
    0 : "Normal",
    1 : "Doubtful",
    2 : "Mild",
    3 : "Moderate",
    4 : "Severe",
}

# Load the model
model = load_model('XceptionModel.hdf5')
print('Model loaded. Check http://127.0.0.1:5000/')

def SeverityChecker(img):
    resize = tf.image.resize(img, (160, 335))
    yhat = model.predict(np.expand_dims(resize / 255, 0))
    yhat = np.argmax(yhat, axis = 1)
    if yhat[0] in dict:
        return dict[yhat[0]]


@application.route("/", methods = ['GET'])
def home():
    return render_template("index.html")

@application.route("/predict", methods = ["GET", "POST"])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = SeverityChecker(cv2.imread(file_path))

        return preds
    return None

if __name__ == '__main__':
    application.run(debug = True)
