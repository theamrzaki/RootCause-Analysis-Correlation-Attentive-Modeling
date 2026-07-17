import pickle
import sys
import types
import numpy as np
import dgl

import dgl
import dgl.heterograph
import sys

# Patch every possible location pickle might look
dgl.heterograph.DGLHeteroGraph = dgl.DGLGraph
if hasattr(dgl, 'DGLGraph'):
    sys.modules['dgl.heterograph'] = dgl.heterograph
    setattr(dgl.heterograph, 'DGLHeteroGraph', dgl.DGLGraph)

#Python 3.9.13, PyTorch 1.12.1, scikit-learn 1.1.2, and DGL 0.9.0 are suggested.


with open("/home/db2003/Desktop/Amr/RootCause-Analysis-Correlation-Attentive-Modeling/datasets/D1_art/D1/samples/train_samples.pkl", "rb") as f:
    train_samples = pickle.load(f)


print(f"Number of training samples: {len(train_samples)}")
print(f"Type: {type(train_samples)}")
print(f"First sample type: {type(train_samples[0])}")
print(f"First sample: {train_samples[0]}")

#pip uninstall -y torch torchvision torchaudio dgl dgl-cu113
#conda uninstall -y pytorch torchvision torchaudio cudatoolkit mkl mkl-service --force
#conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
#python -m pip install dgl-cu113==0.9.0 -f https://data.dgl.ai/wheels/repo.html

