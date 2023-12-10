# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 07:57:53 2023

@author: wrona
"""
from datasets import load_dataset

# Load prepared data from HF
minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")


from datasets import Audio

minds = minds.cast_column("audio", Audio(sampling_rate=16_000))

minds[0]['audio']['sampling_rate']
