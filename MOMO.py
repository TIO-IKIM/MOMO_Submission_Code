"""
@author: Anonymous
"""

# Import relevant packages

import os
import sys

import numpy as np
import pydicom
import json as js
import pandas as pd
import SimpleITK as sitk
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from collections import Counter
from difflib import SequenceMatcher
import functools
import importlib
import keyring, os, signal, time, shutil

import configparser
import ast

import configparser
import MOMO_Backbone as mmb

# Hardcoded imports of the network classes the paper uses for MOMO (Pickle needs these available in the top level namespace)
from ipynb.fs.defs.TransferRes import *
from ipynb.fs.defs.TransferDense import *

# Parse input args
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-s", "--source", dest="source", default=None, help="Use as -s /path/to/study")
parser.add_argument("-c", "--config", dest="config", default="./default_config.ini", help="Use as -c /path/to/config.ini")
parser.add_argument("-a", "--algo", dest="algo", default=12, help="Choose MOMO algorithm to use (0,2,3,4,5,9,12)")
parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="Verbosity. If set, very verbose.")
parser.set_defaults(verbose=False)
args = parser.parse_args()

studydir = args.source
configfile = args.config
algo = int(args.algo)
verbose = args.verbose

if not studydir:
    print(" InputError: No source directory was specified. Exiting.")
    sys.exit(1)

# Load config ini file
print("Loading ini ...")
config_mapfile, config_network, config_networkscript, config_known_metas, config_verbose, config_local, config_split_mode, config_kwargs = mmb.from_config(configfile)

# Load custom script and functions for anyone that wishes to use them
if config_networkscript:
    print("Loading custom network script ("+str(config_networkscript)+") ...")
    custom = importlib.import_module(config_networkscript)
    
print("Done.")

# This is a demo. It makes single predictions for single studies.
# For the full evaluation, you would run mmb.CDtoPrediction against every study
# in whatever dataset you have, and compare with the true labels.

try:
    exitcode, result, error = mmb.CDtoPrediction(data_root = studydir, 
                                                     known_metas = config_known_metas,
                                                     mapfile = config_mapfile, 
                                                     network = config_network, 
                                                     verbose = verbose,
                                                     recoverfrompi = False,
                                                     RecoveredStudyDescription = "None",
                                                     algo = algo,
                                                     split_mode = config_split_mode,
                                                     **config_kwargs)
except:
    print("uh oh")
finally:
    print(exitcode, result, error)

if verbose:
    print("exitcode:", exitcode)
    print("result:", result)
    print("error:", error)

sys.exit(exitcode)