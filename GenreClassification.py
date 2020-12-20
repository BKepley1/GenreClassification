#Music Genre Classification
#Authors: Damian Hupka, Brandon Kepley, Ryan Picariello
#https://github.com/BKepley1/GenreClassification


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, math, librosa
import librosa.display

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D

import sklearn.model_selection as sk
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

#Provides three graphs
def visualize(songLocation):
    #Visualize Waveform
    audio_path = os.getcwd()+"\\Data\\Genres_original\\"+songLocation
    x, sr = librosa.load(audio_path)
    librosa.load(audio_path, sr=None)
    plt.figure(figsize=(16, 5))
    librosa.display.waveplot(x, sr=sr)
    plt.show()

    #Visualize Spectrogram
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14,5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.title('Spectrogram')
    plt.colorbar()
    plt.show()

    #Visualize Mel Spectrogram
    y, sr = librosa.load(audio_path)
    melSpec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    melSpec_db = librosa.power_to_db(melSpec, ref=np.max)
    plt.figure(figsize=(10,5))
    librosa.display.specshow(melSpec_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+1.0f dB')
    plt.title('MelSpectrogram')
    plt.tight_layout()
    plt.show()

#Reads in song files and prepares MFCCs for use in CNN
def prepData(dataPath, numMFCC=20, nFFT=2048, hopLen=512, numSeg=5):
    #Constants as needed for Librosa
    SAMPLE_RATE = 22050
    SONG_DURATION = 30
    SAMP_PER_TRACK = SAMPLE_RATE * SONG_DURATION

    #Samples per segment
    sampPerSeg = int(SAMP_PER_TRACK/numSeg)

                    #Holds the:
    mapping = []    #Index Mapping for each genre
    labels = []     #Label of each audio file
    MFCC = []       #Calculate MFCCS

    #Walks through folders cotaining data
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataPath)):
        if dirpath is not dataPath:
            mapping.append(dirpath.split('\\')[-1])     #Gathers mappings from each folder

        for f in filenames:
            path = os.path.join(dirpath, f)
            if path != os.getcwd() + '\\Data\\genres_original\\jazz\\jazz.00054.wav':   #Skips file as it is corrupted
                signal, sample_rate = librosa.load(path, sr=SAMPLE_RATE)        #Loads audio file

            for d in range(numSeg):
                start = sampPerSeg*d        #First Segment
                finish = start+sampPerSeg   #Last Segment

                # Using Librosa to calculate MFCCS
                mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate,
                                            n_mfcc=numMFCC, n_fft=nFFT, hop_length=hopLen)
                mfcc = mfcc.T   #Transpose of MFCCs

                if len(mfcc) == math.ceil(sampPerSeg/hopLen):   #Filtering out MFCCs with wrong # of features
                    MFCC.append(mfcc.tolist())
                    labels.append(i-1)

    #Conversion to Numpy Array
    MFCC = np.array(MFCC)
    labels = np.array(labels)
    mapping = np.array(mapping)

    return MFCC, labels, mapping


#Separates Data into Train Test and Validation sets using Percentages
def splitData(testSize, valSize):   #Params should be decimal values
    X, Y, z = prepData(dataPath)    #Gets MFCCs, Labels, and mapping

    #Data separation
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=testSize)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=valSize)

    #Adding New Axis as model expects 3D arrays
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_test, X_val, Y_train, Y_test, Y_val, z

#Builds model of inputShape with three ReLu layers using Max Pooling
#Flattens and outputs probabilities using softmax activation
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

#Uses trained model to predict genre of a set of songs
def predict(model, X, y):
    prediction = model.predict(X)
    predictIndex = np.argmax(prediction, axis=1)

    #Uses mapping array to convert prediction to genre
    target = z[y]
    predicted = z[predictIndex]

    #Old Line: Used for single predictions
    #print("Target: {}, Predicted label: {}".format(target, predicted))

    #Outputs confusion Matrix
    confMatrix = confusion_matrix(target, predicted)

    df = pd.DataFrame(confMatrix, z, z)
    plt.matshow(df)
    plt.colorbar()

    #Confusion Matrix in cell labels
    for (x,y), i in np.ndenumerate(df):
        plt.text(x, y, i, va="center", ha="center",
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='.5'))

    #Axix labels
    plt.xticks(np.arange(len(z)), z)
    plt.yticks(np.arange(len(z)), z)
    plt.show()

#Plots Train and Validation accuracy and Error of Model
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
    #Gets path to song files
    #Should work as long as data file structure is as described in readme
    dataPath = os.getcwd() + '\\Data\\genres_original'

    #Uncomment to visualize audio waveform, spectrogram, and mel spectrogram
    #visualize("rock\\rock.00030.wav")

    XTrain, XTest, XVal, YTrain, YTest, YVal, z = splitData(.1, .1)     #Calls splitData to get various sets

    #Defines inputShape for model using Training set shape
    input_shape = (XTrain.shape[1], XTrain.shape[2], 1)
    model = buildModel(input_shape)     #Builds model

    #Compiles model using the optimizer defined below
    optim = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    #Gets history of train and validation accuracy for plotting
    history = model.fit(XTrain, YTrain, validation_data=(XVal, YVal), batch_size=32, epochs=32)

    #Not currently Implemented
    #Would be used to save model to prevent training every run
    # model.save(os.getcwd()+'\\model')
    plotAcc(history)

    #Uses test data to find testing accuracy
    test_loss, test_acc = model.evaluate(XTest, YTest, verbose=2)
    print('\nTest Accuracy:', test_acc)

    #Calls prediction to create confusion matrix using test data
    xToPredict = XTest
    yToPredict = YTest

    predict(model, xToPredict, yToPredict)
