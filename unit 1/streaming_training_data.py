# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 08:34:13 2023

@author: wrona
"""
from datasets import load_dataset
gigaspeech = load_dataset("speechcolab/gigaspeech", "xs", streaming=True)
