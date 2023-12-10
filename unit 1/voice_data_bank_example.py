# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:28:29 2023

@author: wrona
"""

from datasets import load_dataset

# Load prepared data from HF
minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
minds

# See an example
type(minds) #arrow_dataset
example = minds[0]
example
type(example) #dict

# "intent_class" is a label
id2label = minds.features["intent_class"].int2str
id2label(example["intent_class"])
# See all labels: print(minds.features["intent_class"])


# Remove columns
columns_to_remove = ["lang_id", "english_transcription"]
minds = minds.remove_columns(columns_to_remove)
minds

# Audio is made of three things:
example['audio']
'''
- local path to the original .wav
- the single-dimensional numpy array encoding the .wav
- the sampling_rate: 8kHz
'''


# Decode random samples from this data
# Visualize and hear them played from a browser
import gradio as gr


def generate_audio():
    example = minds.shuffle()[0]
    audio = example["audio"]
    return (
        audio["sampling_rate"],
        audio["array"],
    ), id2label(example["intent_class"])


with gr.Blocks() as demo:
    with gr.Column():
        for _ in range(4):
            audio, label = generate_audio()
            output = gr.Audio(audio, label=label)

demo.launch(debug=True)

# See their waveforms
import librosa
import matplotlib.pyplot as plt
import librosa.display

array = example["audio"]["array"]
sampling_rate = example["audio"]["sampling_rate"]

plt.figure().set_figwidth(12)
librosa.display.waveshow(array, sr=sampling_rate)
