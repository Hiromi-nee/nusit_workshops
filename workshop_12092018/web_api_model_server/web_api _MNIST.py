
from __future__ import print_function
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

# sess = tf.Session(config=config)
# set_session(sess)
from keras import backend as K
import numpy as np
import base64
import cv2
import requests
from json import dumps, loads
from keras.models import model_from_json
from flask import Flask, request, jsonify
import time
app = Flask(__name__)

class ClassificationModel():
    def __init__(self):
        K.clear_session() # Clear previous models from memory.
        model_file = "mnist_saved.json"
        weights_file = "mnist_saved_weights.h5" 
        model_json_file = open(model_file, 'r')
        model_json = model_json_file.read()
        model_json_file.close()

        # load model from json file
        self.model = model_from_json(model_json)
        self.model.load_weights(weights_file)
        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.0001),
              metrics=['accuracy'])
        self.model._make_predict_function()

    def classify(self, image):
        y_pred = self.model.predict(np.asarray([image]))
        return y_pred

classification_model = ClassificationModel()

def decode_image(image):
    image = image.encode('utf-8')
    # Decode base64 string
    image = base64.b64decode(image)
    # Read into numpy buffer
    image = np.frombuffer(image, dtype=np.uint8)
    # Decode as cv2 image
    image = cv2.imdecode(image, flags=0)
    return image.reshape(image.shape[0],image.shape[1],1)

@app.route('/classify', methods = ['POST'])
def classify():
    data = request.get_json()
    image = decode_image(data['image'])
    cur_time = time.time()
    retval = classification_model.classify(image)
    elapsed_time = time.time() - cur_time
    print(retval)
    # # print data received
    # print(json_data) 
    
    # # data type
    # print(type(json_data))
    
    # convert to json before sending response
    return jsonify({"prediction":retval[0].tolist(), "elapsed_time":elapsed_time})

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8881)