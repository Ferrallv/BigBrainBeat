# To compile the functions that we use in our project, to keep our presentation orject/program neat.
# Our general imports
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# Base packages used
import keras

# Specific neural network models & layer types
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model

# A wonderful library for audio in python
import librosa

# To display waveforms and spectrograms
import librosa.display

# To listen to tunes
import IPython.display as ipd

# To find bpm
import re

# To get samples
import random

# To resize the spectrogram!
import cv2

# This is the model that we are currently using.
BigBrainBeatv3_phase2 = load_model('BigBrainBeatv3_phase2.h5')

# A function to read in a wav, take the first four seconds, then arrange it to do a prediction
def new_wav_slice(trackpath = "Teleportoise - Epilogue.wav", start = 0):
    """
    This function takes in a wav filepath.
    It then takes the 4 second slice of the wav, starting at the specified second.
    It downsamples the 4 second wav by a quarter. When playing the wav back,
    the specified sampling rate ought to be 11025.
    It reshapes this slice into a matrix to be used for our model.
    It returns the wav in the shape we want for our model.
    """

    # To read in the wav
    wav, sr = librosa.load(trackpath, sr=44100)

    # To normalize the wav
    wav = wav / np.max(wav)

    start = sr*start

    # To take our desired slice
    x_slice = wav[start:(start+(sr*4))]

    # To downsample our x_slice
    x_slice = x_slice[0::4]

    # To reshape into what we want for the model
    matrix = x_slice.reshape(1, 441, 100)

    return matrix

def graph_pred(three_d_mat, model):
    """
    This function takes in a matrix shape(1, 441, 100), and our model.
    The matrix should represent a 4 second 44100hz wav, downsampled to 11025hz and reshaped.
    This function then predicts a bpm based on the range our model has been provided.
    It returns a beat prediction array as well a flattening of the wave matrix to compare in a graph.
    """
    # To get our flattened file for later.
    downsampled_wav = three_d_mat.flatten()


    # A list of possible bpm values
    category_list = list(np.arange(80,161))

    # The liklihoods of different bpm values our model is guessing for the image
    predictions_index = np.argmax(model.predict(three_d_mat))

    print(f"\n\nOur predicted bpm is:{category_list[predictions_index]}")

    # Get the beats per second
    bps = category_list[predictions_index]/60

    # To find how many hertz between each beat
    beat_step = int(11025/bps)

    # In our model and how we process the data at least one beat MUST fall within this range.
    max_check_range = int(11025/(80/60))

    # Setting up our beat illustrater
    beat_pred = np.zeros(44100)

    # Getting the index from where to start the beat inference
    index_start = np.argmax(downsampled_wav[0:max_check_range])

    # To write in where we are inferring the beats should be based on predicition
    for i in range(index_start,44101,beat_step):
        beat_pred[i-1] = 1


    return beat_pred, downsampled_wav

def show_me(trackpath = "WaveBank/two_110.wav", model = BigBrainBeatv3_phase2, seconds = 0):
    """
    This function takes a path to a .wav, runs it through a chosen model, and presents
    the predicted bpm, the downsampled waveform, and the visualization of the matrix used for analysis.

    We will also be able to listen to a stereo 4 second track. The left speaker will play the downsampled audio,
    and the right speaker will be the inferred beats.

    ---
    Argument description:

    trackpath: The pathway to the desired .wav file

    model: The model we will use to predict bpm and infer beat location

    seconds: The time where we want to start our four second slice, in seconds (it can be a fraction)
    ---
    """
    
    # Our function to change our data into the shape we want
    matrix = new_wav_slice(trackpath, start = seconds)

    # To get our waveform and our predicted bpm
    beat_pred, downsampled_wav = graph_pred(matrix, model)

    # To reshape our matrix so that we might visualize it.
    matrix = matrix.reshape(441,100)

    # Plotting our results
    plt.subplots(2,1,figsize=(15,10))

    plt.subplot(2,1,1)
    plt.plot(downsampled_wav)
    plt.plot(beat_pred, "red")

    plt.subplot(2,1,2)
    plt.imshow(matrix.T)

    plt.show();

    return downsampled_wav, beat_pred
