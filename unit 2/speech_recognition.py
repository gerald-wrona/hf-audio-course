# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:24:04 2023

@author: wrona
"""
#%% Perform upsampling to 16kHz

from datasets import Audio
from datasets import load_dataset

# Load prepared data from HF
minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")

# Upsample
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))

#%% Model pipeline

# Load the model
from transformers import pipeline
asr = pipeline("automatic-speech-recognition")

# Feed it one record
example = minds[0]
asr(example["audio"]["array"])

'''
- the model fails on "card", returning "COD"
- so the model underperforms in Australian contexts
- it also doesnt use contextual clues
    - it doesnt know its a banking application
    - i guess thats when prompt engineering is useful, to give
        -- the model the folder name to search within
'''

#%% Model pipeline Deutsch

# Load data and upsample
from datasets import load_dataset
from datasets import Audio

minds = load_dataset("PolyAI/minds14", name="de-DE", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))

# Choose example
example = minds[0]
example["transcription"]

# Check label
id2label = minds.features["intent_class"].int2str
id2label(example["intent_class"])

'''
- the gerne makes it more polite
- without "gerne" the google translation is the same
- there is a bounce, a little cheeriness to the recording
'''

# Run the model
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="maxidl/wav2vec2-large-xlsr-german")
asr(example["audio"]["array"])

'''
- the large model got einzallen instead of einzahlen
- in this context, they will be the same and it's OK,
    -- but similar to before when the model should know from the task context
- english model returns all caps, this returns all lower case
'''