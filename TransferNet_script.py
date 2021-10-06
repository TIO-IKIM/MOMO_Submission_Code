#author: Anonymous

#Imports

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import torchvision.transforms.functional as ttf
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import time

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import random
import itertools
import json as js
import pandas as pd

import DatasetFunctions as df

from collections import Counter
from tqdm import tqdm

import pydicom
import SimpleITK as sitk

#device = torch.device("cpu")
device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")

def representation(SEFN):
    """
    Make a ResNet-usable (224x224x3) representation of an arbitrary input image. There is no center-cropping,
    because we might lose valuable parts of the image (like the Abdomen in a CTTA).
    This function is called by MOMO during evaluation.
    """
    
    # Resample image to correct size
    sitkreader = sitk.ImageSeriesReader()
    sitkreader.SetFileNames(SEFN)
    sitk_image = sitkreader.Execute()
    representation = df.make_representation_from_unknown(current_image = sitk_image, target_size=(224, 224, 224))
    
    # Add a batch dimension
    representation = torch.unsqueeze(torch.Tensor(representation), 0)
    
    # Squeeze to [0,1]
    representation -= representation.min()
    representation /= representation.max()
    
    # Example normalization to ImageNet-mean/std
    ct = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    transformed_representation = ct(representation)
    
    return transformed_representation

def predictor(input_tensor):
    # dummy function, your custom network goes in here
    return None
