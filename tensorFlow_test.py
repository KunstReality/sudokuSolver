from keras.datasets import mnist
from keras.optimizer_v1 import SGD
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.compiler.tf2xla.python.xla import le

from ResNet_architecture import ResNet

#hyper-parameters for fitting model
EPOCHS = 50
INIT_LR = 1e-1
BS = 128

def load_mnist_dataset():

  # load data from tensorflow framework
  ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

  # Stacking train data and test data to form single array named data
  data = np.vstack([trainData, testData])

  # Vertical stacking labels of train and test set
  labels = np.hstack([trainLabels, testLabels])

  # return a 2-tuple of the MNIST data and labels
  return (data, labels)


def train_dataset(train, labels):
  #le = LabelBinarizer()
  #labels = le.fit_transform(labels)

  #Ich denke:
  #soll training data
  trainX = train[0][0]
  #soll training lable
  trainY = train[0][1]
  #soll test data
  testX = train[1][0]
  #soll test lable
  testY = train[1][1]

  counts = labels.sum(axis=0)

  # account for skew in the labeled data
  classTotals = labels.sum(axis=0)
  classWeight = {}

  # loop over all classes and calculate the class weight
  for i in range(0, len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]

  # construct the image generator for data augmentation
  #Data Augmentation
  aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    fill_mode="nearest")

  opt = SGD(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)

  model = ResNet.build(32, 32, 1, len(le.classes_), (3, 3, 3),
                       (64, 64, 128, 256), reg=0.0005)

  model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

  H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS, epochs=EPOCHS,
    class_weight=classWeight,
    verbose=1)
