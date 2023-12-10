# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 12:42:08 2023

@author: wrona
"""

import librosa
array, sampling_rate = librosa.load(librosa.ex("trumpet"))

import matplotlib.pyplot as plt
import librosa.display

plt.figure().set_figwidth(12)
librosa.display.waveshow(array, sr=sampling_rate)