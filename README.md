# BIGBRAINBEAT README
---
BigBrainBeat is a neural network trained on created data to predict the BPM (Beats Per Minute) of a slice of audio. Using the presentation notebook an output will look like this:

```
# To visualize our results, and get what we need to listen!
downsampled_wav1, beat_pred1 = show_me(trackpath = "WaveBank/one_90.wav", model = BigBrainBeatv3_phase1, seconds = 1)

# And to listen to the tune, make sure the sampling rate is correct!
ipd.Audio(data = (downsampled_wav1, beat_pred1), rate =11025)

Our predicted bpm is:90
```
![Downsampled .wav](/images/DownsampledWavWithInferredBeats90BPM.png)

![Visualized .wav](/images/VisualizationOfAudioSlice90BPM.png)

Followed by an audio player that will play the .wav in the left speaker and the inferred beats in the right speaker.

Thank you so much for checking out my project! If you wish to try this at home,
you only need to run the `BigBrainBeat_Presentation.ipynb` with the other files
in the same directory (not including the other `.ipynb`). To change the track,
simply edit the trackpath in the `show_me` function. I encourage you to try your own
recorded samples! Just be sure they are in `.wav` format.

If you are unfamiliar with recording, I recommend https://www.audacityteam.org/
as an excellent open-source starting point.

All of the `.wav` files I used for this project can be found here:
https://drive.google.com/drive/folders/18fEIHUkHM4DmqIjWJiTrqlefFjY0GWpJ?usp=sharing

A walkthrough explanation can be found here: https://ferrallv.github.io/site/project/BigBrainBeat-post/

Please feel free to contact me with any comments, critiques, questions, or ideas!
