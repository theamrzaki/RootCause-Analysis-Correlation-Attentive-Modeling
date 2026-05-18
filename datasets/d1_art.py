import pickle
import sys
import types
import numpy as np
import dgl

with open("/home/db2003/Desktop/Amr/RootCause-Analysis-Correlation-Attentive-Modeling/datasets/D1_art/D1/samples/train_samples.pkl", "rb") as f:
    train_samples = pickle.load(f)


print(f"Number of training samples: {len(train_samples)}")
print(f"Type: {type(train_samples)}")
print(f"First sample type: {type(train_samples[0])}")
print(f"First sample: {train_samples[0]}")