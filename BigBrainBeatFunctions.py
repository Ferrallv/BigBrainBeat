# To compile the functions that we use in our project, to keep our presentation orject/program neat.
# Our general imports
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.plotting import figure, output_notebook, show

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

# This is the model that we are currently using during the testing of these functions.
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

    # To determine the beginning of the wave slice.
    start = int(sr*start)

    # To take our desired slice
    x_slice = wav[start:(start+(sr*4))]

    # To downsample our x_slice to 11025Hz
    x_slice = x_slice[0::4]

    # To reshape into what we want for the model
    matrix = x_slice.reshape(1, 441, 100)

    return matrix

def graph_pred(three_d_mat, model):
    """
    This function takes in a matrix shape(1, 441, 100), and our model.
    The matrix should represent a 4 second 44100hz wav, downsampled to 11025hz and reshaped.
    This function then predicts a bpm based on the range our model has been provided.
    It returns a beat prediction array as well a flattening of the wave matrix to compare to the beats in a graph.
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
        beat_pred[i-1] = downsampled_wav.max()


    return beat_pred, downsampled_wav


# Here we are creating the "pipeline" function. This is essentially an envelope to run the other two functions
# appropriately, while also giving us visualizations of the data.

def show_me(trackpath = "WaveBank/two_110.wav", model = BigBrainBeatv3_phase2, seconds = 0):
    """
    This function takes a path to a .wav, runs it through a chosen model, and presents
    the predicted bpm, the downsampled waveform, and the visualization of the matrix used for analysis.

    With the returned variables we will be able to listen to a stereo 4 second track. The left speaker will play the downsampled audio,
    and the right speaker will be the inferred beats. To do this, we will need to run the IPython.Display.Audio method, using our
    returned variables for the input data, as well as the appropriate sample rate (11025). In future, this feature will likely be embedded in this function.

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

    
    
    """ # This is the previous code with graphs. Below will be our new graphs using bokeh!
    # Plotting our results
    plt.subplots(2,1,figsize=(15,10))

    plt.subplot(2,1,1)
    plt.plot(downsampled_wav)
    plt.plot(beat_pred, "red")

    plt.subplot(2,1,2)
    plt.imshow(matrix.T)

    plt.show();"""
    
    # To make sure it comes out in the notebook
    output_notebook()
    
    # Our first viz, this will show the audio slice with the inferred beats superimposed
    p1 = figure(plot_width=1000, plot_height=400, title="Downsampled Wave with Inferred Beats")
    p1.line(np.arange(0,len(downsampled_wav)), downsampled_wav)
    p1.line(np.arange(0,len(downsampled_wav)), beat_pred, line_color="red", line_width=3)

    # The second viz, which will be the "image" we show the CNN
    p2 = figure(plot_width=1000, plot_height=400, title="Visualization of Audio Slice")
    p2.image(image=[matrix.T], x=0, y=0, dw=441, dh=100, palette="Viridis256")


    show(column(p1,p2));
    
    return downsampled_wav, beat_pred
