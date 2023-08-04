# we use prebuilt mnist model since we only need the digit 0-9
from tensorflow import keras
#from keras.datasets import mnist
from keras.models import load_model
import numpy as np
import cv2

model = load_model('model-OCR.h5')
labels = np.arange(0, 10)

"""def load_mnist():
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

    data = np.vstack([trainData, testData])
    labels = np.hstack([trainLabels, testLabels])

    return (data, labels)"""

def get_prediction(cells):
    predictedt_numbers = []
    prediction = model.predict(cells)
    for weights in prediction:
        index = (np.argmax(weights))
        predictedt_number = labels[index]
        predictedt_numbers.append(predictedt_number)
    if len(predictedt_numbers) > 81:
        return np.zeros((9, 9))
    return np.array(predictedt_numbers).astype('uint8').reshape(9, 9).T



