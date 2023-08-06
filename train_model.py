from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import argparse

# initialize the initial learning rate, number of epochs to train
# for, and batch size
INIT_LR = 1e-3
EPOCHS = 10
BS = 128

class SudokuNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
                # first set of CONV => RELU => POOL layers
        model.add(Conv2D(32, (5, 5), padding="same",
            input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # first set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        # second set of FC => RELU layers
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        # return the constructed network architecture
        return model



def load_mnist():
    print("[INFO] accessing MNIST...")
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

    # add a channel (i.e., grayscale) dimension to the digits
    trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
    testData = testData.reshape((testData.shape[0], 28, 28, 1))

    # scale data to the range of [0, 1]
    trainData = trainData.astype("float32") / 255.0
    testData = testData.astype("float32") / 255.0

    # convert the labels from integers to vectors
    le = LabelBinarizer()
    trainLabels = le.fit_transform(trainLabels)
    testLabels = le.transform(testLabels)

    return (trainData, trainLabels, testData, testLabels, le)

def compile():
    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = Adam(lr=INIT_LR)
    model = SudokuNet.build(width=28, height=28, depth=1, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])
    return model

def evaluate(model, testData, testLabels, le):
    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testData)
    print(classification_report(
        testLabels.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=[str(x) for x in le.classes_]))
    # serialize the model to disk
    print("[INFO] serializing digit model...")
    model.save(args["model"], save_format="h5")

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
        help="path to output model after training")
    args = vars(ap.parse_args())

    trainData, trainLabels, testData, testLabels, le = load_mnist()
    model = compile()
    # train the network
    print("[INFO] training network...")
    H = model.fit(
        trainData, trainLabels,
        validation_data=(testData, testLabels),
        batch_size=BS,
        epochs=EPOCHS,
        verbose=1)
    evaluate(model, testData, testLabels, le)

