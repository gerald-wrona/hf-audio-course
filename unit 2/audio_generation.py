# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:54:59 2023

@author: wrona
"""

#%% Load the model

from transformers import pipeline
pipe = pipeline("text-to-speech", model="suno/bark-small")

# Process text using the pipeline
text = "Ladybugs have had important roles in culture and religion, being associated with luck, love, fertility and prophecy. "
output = pipe(text)

'''
received warning:
    The attention mask and the pad token id were not set.
    As a consequence, you may observe unexpected behavior.
    Please pass your input's `attention_mask` to obtain reliable results.
'''

# Hear the result
from IPython.display import Audio
Audio(output["audio"], rate=output["sampling_rate"])

# Make it sing
song = "♪ In the jungle, the mighty jungle, the ladybug was seen. ♪ "
output = pipe(song)
Audio(output["audio"], rate=output["sampling_rate"])

#%% Music

music_pipe = pipeline("text-to-audio", model="facebook/musicgen-small")
text = "90s rock song with electric guitar and heavy drums"
forward_params = {"max_new_tokens": 512}

output = music_pipe(text, forward_params=forward_params)
Audio(output["audio"][0], rate=output["sampling_rate"])