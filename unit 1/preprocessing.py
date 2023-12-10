# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 07:57:53 2023

@author: wrona
"""
import librosa

#%% Upsampling
from datasets import load_dataset

# Load prepared data from HF
minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")


from datasets import Audio

minds = minds.cast_column("audio", Audio(sampling_rate=16_000))

# Check
minds[0]['audio']['sampling_rate']


#%% Filtering by duration

MAX_DURATION_IN_SECONDS = 20.0


def is_audio_length_in_range(input_length):
    return input_length < MAX_DURATION_IN_SECONDS


# use librosa to get example's duration from the audio file
new_column = [librosa.get_duration(path=x) for x in minds["path"]]
minds = minds.add_column("duration", new_column)

# use 🤗 Datasets' `filter` method to apply the filtering function
minds = minds.filter(is_audio_length_in_range, input_columns=["duration"])

# remove the temporary helper column
minds = minds.remove_columns(["duration"])
minds


#%% Automated feature extraction using Whisper

from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

def prepare_dataset(example):
    audio = example["audio"]
    features = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"], padding=True
    )
    return features


minds = minds.map(prepare_dataset)
minds

# Visualize the result
import numpy as np
import matplotlib.pyplot as plt

example = minds[0]
input_features = example["input_features"]

plt.figure().set_figwidth(12)
librosa.display.specshow(
    np.asarray(input_features[0]),
    x_axis="time",
    y_axis="mel",
    sr=feature_extractor.sampling_rate,
    hop_length=feature_extractor.hop_length,
)
plt.colorbar()


# Tip for multimodal solutions
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("openai/whisper-small")