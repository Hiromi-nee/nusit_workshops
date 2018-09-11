import numpy as np
import base64
import cv2
import requests
from json import dumps
import sys
from pprint import PrettyPrinter
from keras.datasets import mnist
from keras.utils import to_categorical

pp = PrettyPrinter(indent=4)


#172.19.186.4
def classify(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    frame = base64.b64encode(buffer).decode('utf-8')
    # Send to API
    r = requests.post('http://172.19.186.4:8881/classify',
        json={'image': frame, 'frame_no': 0})

    # Print result from API
    #pp.pprint(r.json())
    return r.json()

# get training data
def load_some_data():
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (_, _), (x_test, y_test) = mnist.load_data()

    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    x_test = x_test.astype('float32')
    x_test /= 255
    print(x_test.shape[0], 'test samples')
    y_test_onehot = to_categorical(y_test, 10)

    return x_test, y_test, y_test_onehot

def main():
    idx = 5050 
    x_test, y_test, y_test_onehot = load_some_data()
    
    # shape of image
    print(x_test[idx].shape)

    #classify image
    retval = classify(x_test[idx])
    pp.pprint(retval)

    # get index (class) of highest confidence score
    pred = retval['prediction'].index(max(retval['prediction']))
    print("Actual: %d " % y_test[idx])
    print("Predicted %d" % pred)
    
if __name__ == "__main__":
    main()
