# we use prebuilt mnist model since we only need the digit 0-9
from tensorflow import keras
#from keras.datasets import mnist
from keras.models import load_model
import numpy as np

model = load_model('model-OCR.h5')
img_size = 28

"""def load_mnist():
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

    data = np.vstack([trainData, testData])
    labels = np.hstack([trainLabels, testLabels])

    return (data, labels)"""

def get_prediction(img):
    prediction = model.predict(img)
    return prediction

