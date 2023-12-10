# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 12:55:22 2023

@author: wrona
"""
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

array, sampling_rate = librosa.load(librosa.ex("trumpet"))

D = librosa.stft(array)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure().set_figwidth(12)
librosa.display.specshow(S_db, x_axis="time", y_axis="hz")
plt.colorbar()