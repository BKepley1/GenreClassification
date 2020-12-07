import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, json, math, librosa
import IPython.display as ipd
import librosa.display

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D

import sklearn.model_selection as sk
from sklearn.model_selection import train_test_split

# #Waveform
# audio_path = songLocations[500]
# x, sr = librosa.load(audio_path)
# librosa.load(audio_path, sr=None)
# plt.figure(figsize=(16, 5))
# librosa.display.waveplot(x, sr=sr)
# plt.show()
#
# #Spectrogram
# X = librosa.stft(x)
# Xdb = librosa.amplitude_to_db(abs(X))
# plt.figure(figsize=(14,5))
# librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
# plt.title('Spectrogram')
# plt.colorbar()
# plt.show()
#
# #Mel Spectrogram
# y, sr = librosa.load(audio_path)
# melSpec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
# melSpec_db = librosa.power_to_db(melSpec, ref=np.max)
# plt.figure(figsize=(10,5))
# librosa.display.specshow(melSpec_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
# plt.colorbar(format='%+1.0f dB')
# plt.title('MelSpectrogram')
# plt.tight_layout()
# plt.show()


def prepData(dataPath, numMFCC=20, nFFT=2048, hopLen=512, numSeg=5):
    SAMPLE_RATE = 22050
    SONG_DURATION = 30
    SAMP_PER_TRACK = SAMPLE_RATE * SONG_DURATION

    sampPerSeg = int(SAMP_PER_TRACK/numSeg)
    mapping = []
    labels = []
    MFCC = []

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataPath)):
        if dirpath is not dataPath:
            mapping.append(dirpath.split('\\')[-1])

        for f in filenames:
            path = os.path.join(dirpath, f)
            if path != os.getcwd() + '\\Data\\genres_original\\jazz\\jazz.00054.wav':
                signal, sample_rate = librosa.load(path, sr=SAMPLE_RATE)

            for d in range(numSeg):
                start = sampPerSeg*d
                finish = start+sampPerSeg

                mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=numMFCC, n_fft=nFFT, hop_length=hopLen)
                mfcc = mfcc.T

                if len(mfcc) == math.ceil(sampPerSeg/hopLen):
                    MFCC.append(mfcc.tolist())
                    labels.append(i-1)

    MFCC = np.array(MFCC)
    labels = np.array(labels)
    mapping = np.array(mapping)

    return MFCC, labels, mapping


def splitData(testSize, valSize):
    X, Y, z = prepData(dataPath)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=testSize)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=valSize)

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_test, X_val, Y_train, Y_test, Y_val, z


def buildModel(inputShape):
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=inputShape))
    model.add(keras.layers.MaxPooling2D((3,3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


def predict(model, X, y):
    X = X[np.newaxis, ...]

    prediction = model.predict(X)
    predictIndex = np.argmax(prediction, axis=1)

    target = z[y]
    predicted = z[predictIndex]

    print("Target: {}, Predicted label: {}".format(target, predicted))


def plotAcc(hist):
    fig, axs = plt.subplots(2)
    axs[0].plot(hist.history["accuracy"], label="train accuracy")
    axs[0].plot(hist.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy Eval")

    axs[1].plot(hist.history["loss"], label="train error")
    axs[1].plot(hist.history["val_loss"], label="test error")
    axs[1].set_ylabel("error")
    axs[1].set_xlabel("epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error Eval")

    plt.show()



if __name__ == '__main__':
    dataPath = os.getcwd() + '\\Data\\genres_original'
    XTrain, XTest, XVal, YTrain, YTest, YVal, z = splitData(.25, .2)

    input_shape = (XTrain.shape[1], XTrain.shape[2], 1)
    model = buildModel(input_shape)

    optim = keras.optimizers.Adagrad(learning_rate=0.0001)
    model.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(XTrain, YTrain, validation_data=(XVal, YVal), batch_size=32, epochs=30)

    plotAcc(history)

    test_loss, test_acc = model.evaluate(XTest, YTest, verbose=2)
    print('\nTest Accuracy:', test_acc)
    # songLocations = []
    # genreLabels = []
    # for root, dirs, files in os.walk(dataPath):
    #     for name in files:
    #         filename = os.path.join(root, name)
    #         if filename != '/Data/genres_original/jazz/jazz.00054.wav':  # MAYBE FIX?
    #             songLocations.append(filename)
    #             genreLabels.append(filename.split('\\')[8])
    # print(set(genreLabels))