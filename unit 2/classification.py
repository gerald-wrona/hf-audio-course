# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 08:45:06 2023

@author: wrona
"""

from datasets import load_dataset
from datasets import Audio

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))


from transformers import pipeline

classifier = pipeline(
    "audio-classification",
    model="anton-l/xtreme_s_xlsr_300m_minds14",
)

# Example record
example = minds[0]

# Example classification
classifier(example["audio"]["array"])

# Check intent class actual
id2label = minds.features["intent_class"].int2str
id2label(example["intent_class"])

'''
- the 3% probability for "freeze" ahead of "card_issues" is interesting
- they're both card-related and card is in the transcription
- i wonder if it has to do with speaker tone
- its very tense in response_4.wav whereas response_17.wav is relaxed
'''


# Repeat for another example
example = minds[1]

# Example classification
classifier(example["audio"]["array"])

# Check intent class actual
id2label = minds.features["intent_class"].int2str
id2label(example["intent_class"])
'''
- response_4.wav
- very calm
- the speaker seems to be giving the model misdirection
- "pay for bill" and Bill seems like a person
- the model successfully predicts bill payment despite the two asks
- it might've been trained on this

'''

