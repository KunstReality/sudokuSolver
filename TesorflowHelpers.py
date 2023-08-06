# we use prebuilt mnist model since we only need the digit 0-9
from tensorflow import keras
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np

model = load_model('model-OCR.h5')
labels = np.arange(0, 10)

model2 = load_model('output/digit_classifier.h5')

def get_prediction2(digits):
    predictedt_numbers = []
    for digit in digits:
        if digit is not None:
            roi = img_to_array(digit)
            roi = np.expand_dims(roi, axis=0)
            pred = model2.predict(roi).argmax(axis=1)[0]
            predictedt_numbers.append(pred)
        else:
            predictedt_numbers.append(0)
    if len(predictedt_numbers) > 81:
        return np.zeros((9, 9))
    return np.array(predictedt_numbers).astype('uint8').reshape(9, 9)

def get_prediction(cells):
    predictedt_numbers = []
    prediction = model.predict(cells)
    for weights in prediction:
        index = (np.argmax(weights))
        predictedt_number = labels[index]
        predictedt_numbers.append(predictedt_number)
    if len(predictedt_numbers) > 81:
        return np.zeros((9, 9))
    return np.array(predictedt_numbers).astype('uint8').reshape(9, 9)



