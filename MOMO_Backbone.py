# -*- coding: utf-8 -*-
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

import DatasetFunctions as df
#from ipynb.fs.defs.SCNN_T import *
#from ipynb.fs.defs.TransferRes import *
#from ipynb.fs.defs.TransferDense import *

from pathlib import Path
from tqdm import tqdm
from collections import Counter
from difflib import SequenceMatcher
import functools
import importlib
import keyring, os, signal, time, shutil

import configparser
import ast

# Utility function definitions begin here

def Download(NStudyID, NSeriesID, dest, mode = "all"):
    # In the publication, this is dummy code, because PACS access code will slightly vary
    # depending on the institution and you could tell author and institution from this
    return None

def Reduce(string, **kwargs):
    if isinstance(string, str):
        string = string.replace("ä","ae")
        string = string.replace("ö","oe")
        string = string.replace("ü","ue")
        string = string.replace("Ä","Ae")
        string = string.replace("Ö","Oe")
        string = string.replace("Ü","Ue")
        string = string.replace("ß","ss")
        string = string.replace(" und ","")
        string = string.replace(" and ","")
        string = string.replace(" der ","")
        string = string.replace(" des ","")
        string = string.replace(" of the ","")
        string = string.lower()
        string = ''.join(e for e in string if (e.isalnum() or e == ";"))
        if "reduce_blacklist" in kwargs:
            if not isinstance(kwargs["reduce_blacklist"], list):
                raise TypeError("'reduce_blacklist' kwarg must be a list of strings.")
            for blacklisted_item in kwargs["reduce_blacklist"]:
                string = string.replace(blacklisted_item, "")
        return string
    elif isinstance(string, list):
        ReducedList = []
        for i in range(len(string)):
            ReducedString = Reduce(string[i])
            ReducedList.append(ReducedString)
        return ReducedList

def SubstringMatcher(keys, vals, desc, mapping, meml=4, mrml=6, verbose=False, **kwargs):
    """
    Takes a list of keys and values and matches the values against desc, trying to find the best substring match.
    Exact matches and random matches are treated differently and have different minimum lengths (meml, mrml).
    Order by: Best exact. If no exact, best length. Tiebreak between same length matches, shortest description
    must be right, otherwise we should have been able to match a longer string.
    If a list of priority candidates is given, check whether any of the matches are on the prio list, by
    iterating over the list. If one of the matches is a priority match, that match is returned immediately.
    This implicitly imposes a hierarchy on the priorities (first in list = highest priority).
    """
    matches = []
    sizes = []
    indices = []
    types = []
    str2s = []
    
    # Define match types
    def smallexactmatch(match, s2):
        return (match.size >= meml and s2[match.b-1] == ";" and s2[match.b+match.size] == ";")
    
    def biginsidermatch(match):
        return (match.size >= mrml)
    
    # Match substrings, keep best if better or equal to old best and size sufficient
    string1 = Reduce(desc, **kwargs)
    for idx, item in enumerate(vals):
        string2 = Reduce(item)
            
        # You can never find a longer match than the first, but you can find one that is equally long and it
        # may turn out to be an exact match (or a good one if the first was coincidental). Loop over both
        # strings, advancing only ever far enough that you cannot throw out matches.
        
        ait = 0
        while True:
            #print("ait: ", ait)
            # loop over first string
            if ait == 0:
                prev_a = 0
                prev_size = 0
            else:
                prev_a = first_a+1
            bit = 0
            while True:
                #print("bit:", bit)
                # loop over second string
                if bit == 0:
                    prev_b = 0
                else:
                    prev_b = match.b
                    prev_size = match.size
                match = SequenceMatcher(None, string1, string2, autojunk=False).find_longest_match(prev_a, len(string1), prev_b + prev_size, len(string2))
                #print(string1[match.a:match.a+match.size])
                if bit == 0:
                    first_a = match.a
                if match.size < meml:
                    prev_b = 0
                    break
                if smallexactmatch(match, string2): 
                    matches.append(match)
                    sizes.append(match.size)
                    indices.append(idx)
                    types.append("e")
                    str2s.append(string2)
                elif biginsidermatch(match):
                    matches.append(match)
                    sizes.append(match.size)
                    indices.append(idx)
                    types.append("i")
                    str2s.append(string2)
                bit += 1
            #print(first_a, len(string1))
            #print(matches)
            if first_a == len(string1):
                break
            ait += 1
            
    # Check for exact matches
    b = ""
    c = ""
    
    if matches and "e" in types:
        is_exact = True
        # Collect exact matches
        bms = max([s for i,s in enumerate(sizes) if types[i] == "e"])
        best_i = [i for i,x in enumerate(sizes) if (types[i] == "e" and matches[i].size == bms)]
        best_matches = [x for i,x in enumerate(matches) if (types[i] == "e" and x.size == bms)]
        # Check priority list
        if "ssm_prios" in kwargs:
            for prio in kwargs["ssm_prios"]:
                for im in [(i,m) for i,m in enumerate(matches) if types[i] == "e"]:
                    prio_cn = list(mapping["Internal"]["Code"].keys())[list(mapping["Internal"]["Code"].values()).index(str(prio))]
                    if keys[indices[im[0]]] == str(prio_cn):
                        b = im[1]
                        c = keys[indices[im[0]]]
                        if verbose:
                            print("Priority match found!")
                            print("Matched: "+str(string1[b.a: b.a + b.size])+" in "+string1+" and "+str2s[best_i[0]])
                        prediction = str(prio)
                        if "return_exact" in kwargs:
                            return prediction, is_exact
                        else:
                            return prediction
        # Check if single best match
        if len(best_matches) == 1:
            if verbose:
                print("Found one best exact match.")
            b = best_matches[0]
            c = keys[indices[best_i[0]]]
        # If multiple best, find best out of these
        else:
            if verbose:
                print("Found multiple exact matches.")
            content = [string1[ma.a: ma.a + ma.size] for ma in best_matches]
            if content.count(content[0]) == len(content):
                if verbose:
                    print("Same substring was exactly matched. Finding best class.")
                best_vals = {vals[indices[i]]:indices[i] for i in best_i}
                if verbose:
                    print(best_vals)
                shortest = min(list(best_vals.keys()), key=len)
                if verbose:
                    print(shortest)
                b = best_matches[0]
                c = keys[best_vals[shortest]]
            elif all(x in vals[indices[best_i[0]]] for x in content):
                if verbose:
                    print("Different substrings were exactly matched, but are all of the same class.")
                b = best_matches[0]
                c = keys[indices[best_i[0]]]
            else:
                if verbose:
                    print("Different substrings were exactly matched.")
                    for ma in best_matches:
                        print(str(string1[ma.a: ma.a + ma.size]))
                    print(matches, sizes, indices, types)
                b = ""
                c = ""
    # Check all matches
    elif matches:
        is_exact = False
        # Collect all matches
        bms = max(sizes)
        best_i = [i for i,x in enumerate(sizes) if matches[i].size == bms]
        best_matches = [x for x in matches if x.size == bms]
        # Check priority list
        if "ssm_prios" in kwargs:
            for prio in kwargs["ssm_prios"]:
                for i, current_match in enumerate(matches):
                    prio_cn = list(mapping["Internal"]["Code"].keys())[list(mapping["Internal"]["Code"].values()).index(str(prio))]
                    if keys[indices[i]] == str(prio_cn):
                        b = current_match
                        c = keys[indices[i]]
                        if verbose:
                            print("Priority match found!")
                            print("Matched: "+str(string1[b.a: b.a + b.size])+" in "+string1+" and "+str2s[best_i[0]])
                        prediction = str(prio)
                        if "return_exact" in kwargs:
                            return prediction, is_exact
                        else:
                            return prediction
        # Check if single best match
        if len(best_matches) == 1:
            if verbose:
                print("Found one best match.")
            b = best_matches[0]
            c = keys[indices[best_i[0]]]
        # If multiple best, find best out of these
        else:
            if verbose:
                print("Found multiple matches.")
            content = [string1[ma.a: ma.a + ma.size] for ma in best_matches]
            if content.count(content[0]) == len(content):
                if verbose:
                    print("Same substring was matched. Finding best class.")
                best_vals = {vals[indices[i]]:indices[i] for i in best_i}
                if verbose:
                    print(best_vals)
                shortest = min(list(best_vals.keys()), key=len)
                if verbose:
                    print(shortest)
                b = best_matches[0]
                c = keys[best_vals[shortest]]
            elif all(x in vals[indices[best_i[0]]] for x in content):
                if verbose:
                    print("Different substrings were matched, but are all of the same class.")
                b = best_matches[0]
                c = keys[indices[best_i[0]]]
            else:
                if verbose:
                    print("Different substrings were matched:")
                    for ma in best_matches:
                        print(str(string1[ma.a: ma.a + ma.size]))
                    print(matches, sizes, indices, types)
                b = ""
                c = ""
    else:
        b = ""
        c = ""
                
    # Map our key to its code
    if b:
        if verbose:
            print("Matched: "+str(string1[b.a: b.a + b.size])+" in "+string1+" and "+str2s[best_i[0]])
        prediction = mapping["Internal"]["Code"][str(c)]
        if "return_exact" in kwargs:
            return prediction, is_exact
        else:
            return prediction
    else:
        if verbose:
            print("No good enough match.")
        prediction = None
        if "return_exact" in kwargs:
            return prediction, False
        else:
            return prediction

class meta:
    def __init__(self, name, value):
        self.name = name
        self.value = value

def meta_caller(obj, name:str, primary:tuple, secondary:tuple=None, verbose=False):
    if secondary:
        try:
            return str(obj[primary].value[0][secondary].value), name
        except:
            if verbose:
                print("Secondary "+str(secondary)+" not available, defaulting to primary.")
            if len(obj[primary].value[0]) != 0:
                return str(obj[primary].value), name
            else:
                return None, name
    else:
        return str(obj[primary].value), name
        
def GatherSeriesMetadataFromStudy(data_root, known_metas, verbose=False, **kwargs):
    """
    Walk all folders in data_root, finding all DICOM series along the way, attempting cleanup of garbage files
     (in memory, not on disk), then read DICOM header for the first file in a series, attempting to extract all
     keys specified in known_metas.
    data_root must be a valid path in string format.
    known_metas must be a list of 2-tuples or 3-tuples (or mixed) or a list of lists. The first entry is the
     name of the piece of metadata, the second its DICOM header position. If the piece of metadata is not a
     single entry, but itself a dictionary, a third entry may specify this extra key. Note that several of the
     items in the default known_metas in the .ini are basically always present in a DICOM file and are required.
     These entries should not be removed. Any other entry can be deleted, replaced or new ones added.
    """
    # Find all directories in data_root, which contain at least 1 .dcm file. With good adherence to DICOM standards,
    # this should already be all series.
    DL = df.getDirectoryList(data_root)
    
    SEFNs = []
    SEIDs = []
    meta_dict = {}
    
    # Take all paths, and check if they contain series (they should)
    for seriesdir in DL:
        
        # If it is an image series, files should be readable as a series
        sitkreader = sitk.ImageSeriesReader()

        # Check if there is a DICOM series in the dicom_directory
        series_IDs = sitkreader.GetGDCMSeriesIDs(seriesdir)
        if verbose:
            print ("Loading dicom folder %" + seriesdir)
            print ("Detected "+str(len(series_IDs))+" distinct series. Loading files ...")
        
        for idx, ID in enumerate(series_IDs):
            try:
                # Get all file names
                series_file_names = sitkreader.GetGDCMSeriesFileNames(seriesdir, series_IDs[idx])
                if verbose:
                    print(str(len(series_file_names))+" files in series. Attempting cleanup if necessary ...")
                file_sizes = []

                # Try cleaning out garbage from series
                for file in series_file_names:
                    filereader = sitk.ImageFileReader()
                    filereader.SetFileName(file)
                    tmp = filereader.Execute()
                    size = tmp.GetSize()
                    origin = tmp.GetOrigin()
                    spacing = tmp.GetSpacing()
                    file_sizes.append((size[0], size[1]))
                size_hist = Counter(file_sizes)
                wanted_size = max(size_hist, key=size_hist.get)
                series_file_names = [name for idx, name in enumerate(series_file_names) if file_sizes[idx] == wanted_size]
                if verbose:
                    print("Cleanup complete. "+str(len(series_file_names))+" files remain in series.") 

                # Try to grab relevant data, setting defaults
                meta_read = pydicom.filereader.dcmread(str(series_file_names[0]), stop_before_pixels=True)
                
                for item in known_metas:
                    try:
                        if len(item) == 2:
                            v, cn = meta_caller(obj=meta_read, name=item[0], primary=item[1], verbose=verbose)
                            m = meta(cn, v)
                        elif len(item) == 3:
                            v, cn = meta_caller(obj=meta_read, name=item[0], primary=item[1], secondary=item[2], verbose=verbose)
                            m = meta(cn, v)
                        else:
                            raise ValueError("Falsely formatted known metadata: "+str(item))    
                        if verbose:
                            print("Key read successfully: "+str(item[1])+".")
                            print(cn)
                            print(v)
                        
                    except KeyError as k:
                        if verbose:
                            print("Key not found in DICOM header: "+str(k)+", skipping key.")
                        # Send along an empty string if you find nothing, because we need a string, not a NoneType later
                        m = meta(item[0], "")
                        
                    except ValueError as v:
                        if verbose:
                            print("Bad metadata request: "+str(v))        
                    
                    if not item[0] in list(meta_dict.keys()):
                        meta_dict[item[0]] = [m]
                    else:
                        meta_dict[item[0]].append(m)
                
                SEFNs.append(series_file_names)
                SEIDs.append(ID)
                
            except Exception as e:
                if verbose:
                    print("An error occured, moving on to next series. (Error: "+str(e)+")")
    
    return meta_dict, SEFNs, SEIDs

class vote_rule():
    """
    Instantiate a voting rule according to the following syntax (must be provided as one string):
    "a+b=c" <=> if a and b in list, replace all a and b with c (more additions are allowed, but only one = sign)
    "a-b=c" <=> if a and not b in list, replace all a and b with c (more subs allowed, but only one = sign)
    Rule 1 and 2 may be concatenated infinitely as "a+b-c+d=e", but cannot start with a "-"-sign
    "a>b"   <=> if a and b in list, replace all b with a (only one argument is allowed on each side)
    "a!"    <=> if a in list, a is absolute (only one argument is allowed)
    Any vote of rtype "!" will cause the voting process to exit, votes being the results of the applied rule
    "a!+b"  <=> if a and b in list, a is absolute (more additions are allowed, but only one absolute value)
    "a!-b"  <=> if a and not b in list, a is absolute (more subtractions are allowed, but only one absolute value)
    Rule 5 and 6 may be concatenated infinitely as "a!+b-c+d-e"
    
    This class does not check whether the rules contain valid strings, that one is up to the user!
    All rules will be applied in the order they are entered in. If any rules interact, be aware of this.
    Application of any remaining rules is skipped if a rule with an absolute target is found to apply.
    self.is_absolute is only returned as True for a rule, if the rule's conditional is True and its rtype is '!'
    """
    def __init__(self, rule_string: str):
        # Defaults
        self.string = rule_string
        self.is_absolute = False
        self.conditional = True
        self.target = None
        self.rtype = None
    
        # Extract operators, substrings
        pos = [-1] + [i for i, c in enumerate(self.string) if not c.isalnum()] + [len(self.string)]
        self.operators = [c for c in self.string if not c.isalnum()]
        self.substrings = [self.string[pos[j]+1:pos[j+1]] for j, op in enumerate(pos[:-1]) if self.string[pos[j]+1:pos[j+1]]]
        if self.operators[-1] == "=":
            self.rtype = "="
            self.target = self.substrings[-1]
        if self.operators[0] == ">" and len(self.operators) == 1:
            self.rtype = ">"
            self.target = self.substrings[0]
        if "!" in self.operators:
            self.rtype = "!"
            self.target = self.substrings[0]
        
        # Malformed rules cause exceptions
        if any([True for op in self.operators if op not in ["+","-","!",">","="]]):
            raise SyntaxError('Invalid rule operator specified: Operators must be ["+","-","!",">","="]')
        if len(self.operators) == 0:
            raise SyntaxError('No rule operators specified, invalid rule.')
        if self.operators[-1] == "=" and any([True for op in self.operators[:-1] if op != "+"]):
            raise SyntaxError('Rule operators are not rules compliant.')
        if ">" in self.operators and len(self.operators) > 1:
            raise SyntaxError('Rule operators are not rules compliant.')
        if "!" in self.operators and len(self.operators) > 1:
            if "!" in self.operators[1:]:
                raise SyntaxError('Rule operators are not rules compliant.')
        
    def apply_to(self, votes: list):
        # Default conditional and is_absolute
        self.conditional = True
        self.is_absolute = False
        
        # Rules 1 and 2
        if self.operators[-1] == "=":
            self.rtype = "="
            self.is_absolute = False
            self.target = self.substrings[-1]
            for k, op in enumerate(self.operators):
                if k == 0:
                    self.conditional &= self.substrings[k] in votes
                    last_op = op
                else:
                    if last_op == "+":
                        self.conditional &= self.substrings[k] in votes
                        last_op = op
                    elif last_op == "-":
                        self.conditional &= not self.substrings[k] in votes
                        last_op = op
        
        # Rule 3
        if self.operators[0] == ">" and len(self.operators) == 1:
            self.rtype = ">"
            self.is_absolute = False
            self.conditional &= self.substrings[0] in votes and self.substrings[1] in votes
            self.target = self.substrings[0]
            
        # Rule 4
        if "!" in self.operators:
            self.rtype = "!"
            self.conditional &= self.substrings[0] in votes
            self.target = self.substrings[0]
            # Rules 5 and 6
            if len(self.operators) > 1:
                for k, op in enumerate(self.operators[1:]):
                    if op == "+":
                        self.conditional &= self.substrings[k+1] in votes
                    elif op == "-":
                        self.conditional &= not self.substrings[k+1] in votes
            if self.conditional:
                self.is_absolute = True

        a = []
        for vote in votes:
            if vote:
                if vote in self.substrings:
                    if self.conditional:
                        a.append(self.target)
                    else:
                        a.append(vote)
                else:
                    a.append(vote)
            else:
                a.append(vote)
                
        revised_votes = [self.target if self.conditional and vote and (vote in self.substrings or self.rtype == "!") else vote for vote in votes]
        #print(self.string, self.conditional)
        return revised_votes, self.is_absolute, self.target
    
def from_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    # positional
    mapfile = config["positional"].get("mapfile", "./MCMapping.json")
    network = config["positional"].get("network", "./eval_network_all.pth")
    networkscript = config["positional"].get("networkscript", "")
    known_metas = ast.literal_eval(config["positional"].get("known_metas", []))
    if not known_metas:
        raise ConfigError("known_metas must at least contain Procedure Code, Study Description and Series Modality keys.")
    # optional
    verbose = ast.literal_eval(config["optional"].get("verbose", False))
    local = ast.literal_eval(config["optional"].get("local", True))
    split_mode = ast.literal_eval(config["optional"].get("split_mode", False))
    # kwargs
    config_kwargs = {}
    mapfiles = ast.literal_eval(config["keywordargs"].get("mapfiles", {}))
    config_kwargs["mapfiles"] = mapfiles
    
    networks = ast.literal_eval(config["keywordargs"].get("networks", {}))
    config_kwargs["networks"] = networks
    
    custom_representation = ast.literal_eval(config["keywordargs"].get("custom_representation", False))
    config_kwargs["custom_representation"] = custom_representation
    
    custom_predictor = ast.literal_eval(config["keywordargs"].get("custom_predictor", False))
    config_kwargs["custom_predictor"] = custom_predictor
    
    vote_rules = ast.literal_eval(config["keywordargs"].get("vote_rules", []))
    classed_rules = [vote_rule(rule) for rule in vote_rules]
    config_kwargs["vote_rules"] = classed_rules
    
    network_vote_rules = ast.literal_eval(config["keywordargs"].get("network_vote_rules", False))
    config_kwargs["network_vote_rules"] = network_vote_rules
    
    remapped_modalities = ast.literal_eval(config["keywordargs"].get("remapped_modalities", {}))
    config_kwargs["remapped_modalities"] = remapped_modalities
    
    minmatch_length = ast.literal_eval(config["keywordargs"].get("minmatch_length", {}))
    config_kwargs["minmatch_length"] = minmatch_length
    
    blacklist = ast.literal_eval(config["keywordargs"].get("blacklist", []))
    config_kwargs["blacklist"] = blacklist
    
    reduce_blacklist = ast.literal_eval(config["keywordargs"].get("reduce_blacklist", []))
    config_kwargs["reduce_blacklist"] = reduce_blacklist
    
    ssm_prios = ast.literal_eval(config["keywordargs"].get("ssm_prios", []))
    config_kwargs["ssm_prios"] = ssm_prios
    
    no_network = ast.literal_eval(config["keywordargs"].get("no_network", False))
    config_kwargs["no_network"] = no_network
    
    netw_conf_threshold = float(config["keywordargs"].get("netw_conf_threshold", 0.0))
    config_kwargs["netw_conf_threshold"] = netw_conf_threshold
    
    a12_gmw = float(config["keywordargs"].get("a12_gmw", 1.5))
    config_kwargs["a12_gmw"] = a12_gmw
    
    a12_wmw = float(config["keywordargs"].get("a12_wmw", 0.5))
    config_kwargs["a12_wmw"] = a12_wmw
    
    return mapfile, network, networkscript, known_metas, verbose, local, split_mode, config_kwargs

# Algorithmic function definitions begin here

def PredictSeriesWithNetwork(STModa, SEDesc, SEModa, SEFN, SEID, mapfile, network, verbose=False, **kwargs):
    """
    Given metadata, a mapfile, a network for evaluation and potential kwargs, return the eligibility of the
     prediction as a vote, the probability of the prediction and the prediction itself. Predictions should be
     None, or one of the classes supplied in the mapfile.
    If "blacklist" is supplied as a kwarg, series containing a blacklisted substring in their description,
     lose eligibility and are not evaluated.
    """
    
    # Valid modalities
    with open(mapfile) as json_file:
        classmapping = js.load(json_file)
    valid_modas = []
    for moda in list(classmapping["Internal"]["Moda"].values()):
        if moda not in valid_modas and moda==moda:
            valid_modas.append(moda)
    
    # Default vote value
    netw_vote = None
    # If series on blacklist, disallow voting
    if "blacklist" in kwargs:
        if SEDesc:
            if any(substring in Reduce(SEDesc, **kwargs) for substring in kwargs["blacklist"]):
                if verbose:
                    print("Removed series "+str(SEDesc)+" from network prediction because it matches a blacklisted item.")
                eligibility = 0
                probability = 0
                return eligibility, probability, netw_vote
    if SEModa in valid_modas:
        eligibility = 1
    else:
        eligibility = 0
        probability = 0
        return eligibility, probability, netw_vote
    
    # load image
    if "custom_representation" in kwargs and kwargs["custom_representation"] == True:
        if verbose:
            print("Making tensor representation using custom representation function ...")
        try:
            tensor_representation = kwargs["custom_script"].representation(SEFN)
        except Exception as e:
            if verbose:
                print("Error loading image using custom function: ", e)
            return 0, 0, None
    else:
        try:
            sitkreader = sitk.ImageSeriesReader()
            sitkreader.SetFileNames(SEFN)
            sitk_image = sitkreader.Execute()
            representation = df.make_representation_from_unknown(current_image = sitk_image, target_size=(512,512,512), verbose=verbose)
        except Exception as e:
            if verbose:
                print("Error loading image: ", e)
            return 0, 0, None

        # add batch dimension to image, normalize
        tensor_representation = torch.unsqueeze(torch.Tensor(representation), 0)
        tensor_representation -= tensor_representation.min()
        tensor_representation /= tensor_representation.max()
    
    with torch.no_grad():
        # load network
        net = torch.load(network)
        # set to eval mode
        net.eval()
        # collect results
        if "custom_predictor" in kwargs and kwargs["custom_predictor"] == True:
            if verbose:
                print("Making prediction using custom prediction function ...")
            try:
                class_prediction, probability = kwargs["custom_script"].predictor(tensor_representation)
            except Exception as e:
                if verbose:
                    print("Error in custom network predictor function: ", e)
                return 0, 0, None
        else:
            logits = net(tensor_representation)
            if verbose:
                print(logits)
            lsm = torch.nn.functional.log_softmax(logits, dim=1)
            sm = torch.nn.functional.softmax(logits, dim=1)
            prediction = lsm.data.max(1, keepdim=True)[1][0].item()
            probability = sm.data.max(1, keepdim=True)[0][0].item()

            # map prediction to readable class name, if PETMap exists (because split_mode = True), use it
            if "PETMap" in list(classmapping.keys()) and (STModa == "PT" or STModa == "MRPET"):
                class_prediction = classmapping["PETMap"][classmapping["Internal"]["Code"][str(prediction)]]
            else:
                class_prediction = classmapping["Internal"]["Code"][str(prediction)]
    
    # return result
    if np.isnan(probability):
        eligibility = 0
    if verbose:
        print(eligibility, probability, class_prediction)
    return eligibility, probability, class_prediction

def PredictSeries(meta_dict, sidx, STModa, mapping, verbose=False, **kwargs):
    
    # grab series modality
    SEModa = meta_dict["Series Modality"][sidx].value
    
    # remap if needed
    if "remapped_modalities" in kwargs and SEModa in list(kwargs["remapped_modalities"].keys()):
        SEModa = str(kwargs["remapped_modalities"][str(SEModa)])
    
    # Unpack every non-critical meta information from meta_dict to matchables
    matchables = []
    matchables_names = []
    for key in sorted(list(meta_dict.keys())):
        # If the key is not one of the critical keys, unpack it, unless it is None (because it was not found)
        if not key in ["Procedure Code", "Study Description", "Series Modality"]:
            if sidx < len(meta_dict[key]):
                if meta_dict[key][sidx]:
                    matchables.append(meta_dict[key][sidx].value)
                    matchables_names.append(meta_dict[key][sidx].name)
                else:
                    matchables.append(None)
                    matchables_names.append(key)
            else:
                matchables.append(None)
                matchables_names.append(key)
                
    votes = []
    
    # If PET study, disallow voting for all non-PETs
    #if STModa == "PT" or STModa == "MRPET":
    #    if SEModa == "PT":
    #        eligibility = 1
    #    else:
    #        eligibility = 0
    #        votes += len(matchables) * [None]
    #        return eligibility, votes
    # If other study, allow voting for all series with at least one metadata entry
    if SEModa and any(matchables):
        eligibility = 1
    else:
        eligibility = 0
        votes += len(matchables) * [None]
        return eligibility, votes
    
    ####################################################################################################
    decider = "(Longest) Reduced Substring Match (matchable metadata -> Desc + Alts) with known modality"

    # Grab all descriptions from mapping which are legal, based on known modality
    if STModa == "MRPET":
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
    elif STModa == "MR":
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
    elif not STModa:
        # Making sure to exclude the nan variant from the JSON
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if key==key]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if val==val]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if key==key]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if val==val]
    else:
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
    
    # If there is only one item in keys, we do not need to match anything, this is the only possibility
    # Typically, this should only ever apply to MG, but who knows. If this is incorrect, then the mapping
    # needs to be adjusted.
    desclen = len(keys)/2
    if desclen == 1:
        stdesc_prediction = mapping["Internal"]["Code"][str(keys[0])]
        if verbose:
            print("Modality has only one entry in mapping, this should be the correct class.")
        return stdesc_prediction, decider
    
    # Add the modality to all keywords, and once at the end for description, add PT as PETCT
    if STModa == "PT":
        ML = ["PT", "PET CT", "CT PET"]
    elif STModa == "MR":
        ML = ["MRT", "MR"]
    elif STModa == "MRPET":
        ML = ["MRT PET", "PET MRT", "PET MR", "MR PET"]
    else:
        ML = [STModa]
    for idx, alts in enumerate(vals):
        if idx < desclen:
            vals[idx] = str(";")+";".join([vals[idx]+str(";")+vals[idx]+M for M in ML])+str(";")
        else:
            vals[idx] = str(";")+";".join(";".join([item+M+str(";")+M+item+str(";")+item for item in alts.split(";")]) for M in ML)+str(";")
        
    # Substring matching
    if "minmatchlength" in kwargs:
        meml = kwargs["minmatchlength"]["series"][0]
        mrml = kwargs["minmatchlength"]["series"][1]
    else:
        meml = 3
        mrml = 6
        
    for i, matchable in enumerate(matchables):
        if matchable:
            if verbose:
                print(str(matchables_names[i])+" exists, try to match substrings")
            vote = SubstringMatcher(keys = keys,
                                    vals = Reduce(vals),
                                    desc = matchable,
                                    meml = meml, 
                                    mrml = mrml, 
                                    mapping = mapping, 
                                    verbose = verbose, 
                                    **kwargs)
            votes.append(vote)
        else:
            if verbose:
                print(str(matchables_names[i])+" is 'None', not matching.")
            votes.append(None)
    ####################################################################################################
    
    return eligibility, votes

def PredictStudy_5(meta_dict, mapfile, network, verbose=False, local=True, split_mode=False, **kwargs):
    '''
    meta_dict must be supplied either by extraction from some database query, or, more likely (if local=True),
     from a call to GatherSeriesMetadataFromStudy. file_names and series_ids are supplied in the same manner.
     Some entries in the meta_dict are required (see function definition of GatherSeriesMetadataFromStudy).
     
    If local=False, must supply NStudyID and NSeriesIDs as kwargs (string, list of strings). Note that these
     IDs are the ones for querying in the database (PACS), not the ones saved in the header (these might be
     differnt if the data is deidentified prior to being transferred from the database to the local machine).
    If split_mode=True, must supply dicts, named "mapfiles" and "networks", where the keys are the modalities
     and the values paths to .json maps and .pth networks (for mapfiles and networks respectively). 
    If vote_rules is supplied as a kwarg, it must conform to the rules set in the class definition of vote_rule.
     vote_rules must be a list of vote_rule instances, such as [vote_rule("CTAB+CTT=CTTA"), vote_rule("CTH!")].
    If remapped_modalities is supplied as a kwarg, it must be a dictionary, where the keys are the modality that
     is thrown out and the values are the modalities we replace them with (example: {"DX": "CR"} would treat any
     DX (Direct Radiography series) as CR (Computed Radiography series) during study prediction). Generally this
     is not recommended except for debugging, but your mileage may vary.
    If minmatchlength is supplied as a kwarg, it must be a dictionary containing the keys "study" and "series"
     with the values tuple(minimum length of exact matches, minimum length of random matches). If any is supplied,
     all must be supplied. The defaults are (4, 6) and (3, 6).
    '''
    
    # Unpack critical metadata
    try:
        Code = meta_dict["Procedure Code"][0].value
        STDesc = meta_dict["Study Description"][0].value
        SEModas = [item for item in meta_dict["Series Modality"]]
    except KeyError as k:
        print("meta_dict is missing critical metadata key: "+str(k)+". Study will not be predicted.")
        print("If this error shows up constantly, the key was probably excluded in the config.ini.")
        print("If this is the case, consider restoring it.")
        raise
        
    # Valid modalities
    with open(mapfile) as json_file:
        mapping = js.load(json_file)
    valid_modas = []
    for moda in list(mapping["Internal"]["Moda"].values()):
        if moda not in valid_modas and moda==moda:
            valid_modas.append(moda)
            
    # Remapped modalities (pretend all series of type X are actually type Y, generally only for debugging)
    if "remapped_modalities" in kwargs:
        if verbose:
            print("Remapping modalities ...")
        SEModas = [SEM if SEM.value not in kwargs["remapped_modalities"] else meta("Series Modality", kwargs["remapped_modalities"][str(SEM.value)]) for SEM in SEModas]

    # Rule-based approach to Modality of Study
    if any(SEModas):
        try:
            tmp = [m.value for m in SEModas if m.value in valid_modas]
            hist = (Counter(tmp))
            STModa = max(hist, key=hist.get)
        except:
            STModa = ""
    else:
        STModa = ""
    if any(x for x in SEModas if x.value == "PT"):
        if STModa == "CT":
            STModa = "PT"
        elif STModa == "MR":
            STModa = "MRPET"
            
    # Classify study as unknown if the modality is entirely unmapped
    if not STModa:
        try:
            tmp = [m.value for m in SEModas if m.value]
            hist = (Counter(tmp))
            STModa = max(hist, key=hist.get)
        except:
            pass
        if not (STModa in valid_modas):
            prediction = "UNKNOWN-"+str(STModa)
            decider = "Unmapped modality provided"
            return prediction, decider
        else:
            prediction = "UNKNOWN"
            decider = "No modality provided"
            return prediction, decider
    
    ####################################################################################################
    # Layer 1
    ####################################################################################################
    if verbose:
        print("Trying to match procedure code.")
    decider = "Code, Full Match"
    try:
        c = list(mapping["Internal"]["Code"].keys())[list(mapping["Internal"]["Code"].values()).index(str(Code))]
        prediction = mapping["Internal"]["Code"][str(c)]
        if verbose:
            print("Full match for procedure code found.")
        return prediction, decider
    except:
        # No match for "Code", try Multiclass
        try:
            prediction = mapping["Internal"]["Multiclass"][str(Code)]
            if verbose:
                print("Full match for procedure code found.")
            return prediction, decider
        except:
            # No matches
            if verbose:
                print("No exact match for Procedure Code in mapping.")
    ####################################################################################################
    # Layer 2
    ####################################################################################################
    if verbose:
        print("Trying to match study description.")
    decider = "Reduced Study Description, Full Match"
    try:
        c = Reduce(list(mapping["Internal"]["Desc"].keys()))[Reduce(list(mapping["Internal"]["Desc"].values())).index(str(Reduce(STDesc)))]
        prediction = mapping["Internal"]["Code"][str(c)]
        # Only accept full match if modality is correct (unless no modality is given, in which case, accept all)
        if mapping["Internal"]["Moda"][str(c)] == STModa or not STModa:
            if verbose:
                print("Full match for Study Description found.")
            return prediction, decider
        else:
            if verbose:
                print("Exact match for study description in mapping, but wrong modality.")
    except:
        # No exact match
        if verbose:
            print("No exact match for study description in mapping.")
    ####################################################################################################
    # Layer 3
    ####################################################################################################
    if verbose:
        print("Trying to match substrings from collected metadata.")
    decider = "Voting (1 vote each best substring match over all collected metadata)"
    
    # Grab all descriptions from mapping which are legal, based on known modality
    # If modality is unknown or not one of the main ones, allow all modalities
    # Grab all descriptions from mapping which are legal, based on known modality
    if STModa == "MRPET":
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
    elif STModa == "MR":
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
    elif not STModa:
        # Making sure to exclude the nan variant from the JSON
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if key==key]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if val==val]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if key==key]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if val==val]
    else:
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
    
    # If there is only one item in keys, we do not need to match anything, this is the only possibility
    # Typically, this should only ever apply to MG, but who knows. If this is incorrect, then the mapping
    # needs to be adjusted.
    desclen = len(keys)/2
    if desclen == 1:
        stdesc_prediction = mapping["Internal"]["Code"][str(keys[0])]
        if verbose:
            print("Modality has only one entry in mapping, this should be the correct class.")
        return stdesc_prediction, decider
    
    # Add the modality to all keywords, and once at the end for description, add PT as PETCT
    if STModa == "PT":
        ML = ["PT", "PET CT", "CT PET"]
    elif STModa == "MR":
        ML = ["MRT", "MR"]
    elif STModa == "MRPET":
        ML = ["MR PET", "PET MR", "MRT PET", "PET MRT"]
    else:
        ML = [STModa]
    for idx, alts in enumerate(vals):
        if idx < desclen:
            vals[idx] = str(";")+";".join([vals[idx]+str(";")+vals[idx]+M for M in ML])+str(";")
        else:
            vals[idx] = str(";")+";".join(";".join([item+M+str(";")+M+item+str(";")+item for item in alts.split(";")]) for M in ML)+str(";")

    # Substring matching
    if "minmatchlength" in kwargs:
        meml = kwargs["minmatchlength"]["study"][0]
        mrml = kwargs["minmatchlength"]["study"][0]
    else:
        meml = 4
        mrml = 6
    stdesc_prediction, is_exact = SubstringMatcher(keys = keys,
                                                   vals = Reduce(vals),
                                                   desc = STDesc, 
                                                   meml = meml, 
                                                   mrml = mrml, 
                                                   mapping = mapping, 
                                                   verbose = verbose, 
                                                   return_exact = True, 
                                                   **kwargs)
            
    # Instant hits - If we score an exact match in the study description, there is no reason to continue and
    # we immediately return the value
    if is_exact:
        decider = "Best substring match for Study Description"
        if verbose:
            print("Study description had exact match, this is our best guess.")
        return stdesc_prediction, decider
    
    ####################################################################################################
    # Layer 4
    ####################################################################################################
    # Series Voting starts here, try matching as above and then vote by simple majority for a prediction
    if verbose:
        print("Beginning series voting...")
    decider = "Series Voting"

    preliminary_votes = []
    mask = []
    for i in range(len(meta_dict["Procedure Code"])):
        eligibility, votes = PredictSeries(meta_dict = meta_dict,
                                           sidx = i,
                                           STModa = STModa,
                                           mapping = mapping,
                                           verbose = verbose,
                                           **kwargs)
        preliminary_votes.extend(votes)
        mask.append(eligibility)
    
    if verbose:
        print("=Preliminary votes=")
        kc = 0
        for key in sorted(list(meta_dict.keys())):
            if not key in ["Procedure Code", "Study Description", "Series Modality"]:
                print(str(key)+" Votes:")
                print(preliminary_votes[kc::(len(meta_dict)-3)])
                kc += 1
        print("Is eligible:")
        print(mask)
        
    # Apply voting rules (if a rule establishes an absolute, exit rules application. (All legal votes have been set
    # to that absolute value anyway, meaning we can just count votes as we normally would and the absolute wins.)
    if "vote_rules" in kwargs:
        if verbose:
            print("Applying voting rules...")
        combined_votes = preliminary_votes
        for rule in kwargs["vote_rules"]:
            combined_votes, r_is_absolute, r_target = rule.apply_to(combined_votes)
            if r_is_absolute:
                break
        if verbose:
            print("=Corrected votes=")
            kc = 0
            for key in sorted(list(meta_dict.keys())):
                if not key in ["Procedure Code", "Study Description", "Series Modality"]:
                    print(str(key)+" Votes:")
                    print(combined_votes[kc::(len(meta_dict)-3)])
                    kc += 1
            print("Is eligible:")
            print(mask)
    
    # Count all votes, if the vote is not None and the voting series was deemed eligible
    winners = []
    try:
        vote_hist = Counter([vote for i,vote in enumerate(combined_votes) if (mask[int(np.floor(i/(len(meta_dict)-3)))] == 1 and vote)])
        most_votes = max(vote_hist.values())
        winners = [key for i, key in enumerate(list(vote_hist.keys())) if list(vote_hist.values())[i] == most_votes]
        if verbose:
            print(vote_hist, most_votes, winners)
    except:
        if verbose:
            print("Cannot count series votes.")
    if len(winners) == 0:
        if verbose:
            print("No votes were cast. No eligible series ("+str(valid_modas)+") found or they were no help during substring matching.")
    if len(winners) == 1:
        return winners[0], decider
    ####################################################################################################
    # Layer 5 (Network)
    ####################################################################################################
    else:
        # Network tiebreak
        decider = "Neural Network"
        if "no_network" in kwargs:
            if kwargs["no_network"] == True:
                if verbose:
                    print("Network predictions disabled, no vote.")
                prediction = "UNKNOWN-"+str(STModa)
                return prediction, decider
        if verbose:
            print("Feeding data into network for tiebreak.")
            
        # If predictions are made locally, the files need not be downloaded - non-local files must be acquired
        # first, else the network has nothing to evaluate on
        if not local:
            if verbose:
                print("Acquiring data ...")
            # Remove potential leftovers
            DLPath = './tempeval/'
            shutil.rmtree(DLPath)
            os.makedirs(DLPath, exist_ok=True)
            # Download files
            for k in range(len(kwargs["NSeriesIDs"])):
                try:
                    Download(NStudyID = kwargs["NStudyID"],
                             NSeriesID = kwargs["NSeriesIDs"][k],
                             dest = DLPath,
                             mode = "all")
                except Exception as e:
                    if verbose:
                        print("Download failed. Errmsg: "+str(repr(e)))
        
            _, file_names, series_ids = GatherSeriesMetadataFromStudy(data_root = str(DLPath),
                                                                      known_metas = kwargs["known_metas"],
                                                                      verbose = verbose)
        else:
            file_names = kwargs["file_names"]
            series_ids = kwargs["series_ids"]
            
        # If split mode, figure out network/mapfile, then predict, else predict with default network/mapfile
        if split_mode:
            mapfile = kwargs["mapfiles"][str(STModa)]
            network = kwargs["networks"][str(STModa)]
            
        netw_mask = []
        netw_probs = []
        netw_votes = []
        
        for m in range(len(series_ids)):
            try:
                # Default series descriptions for emergencies
                if not "Series Description" in list(meta_dict.keys()):
                    meta_dict["Series Description"] = []
                    for i in range(len(meta_dict["Series Modality"])):
                        meta_dict["Series Description"].append(meta("Series Description", "None"))
                eligibility, probability, prediction = PredictSeriesWithNetwork(STModa = STModa,
                                                                                SEDesc = meta_dict["Series Description"][m].value,
                                                                                SEModa = SEModas[m].value,
                                                                                SEFN = file_names[m],
                                                                                SEID = series_ids[m],
                                                                                mapfile = mapfile,
                                                                                network = network,
                                                                                verbose = verbose,
                                                                                **kwargs)
                netw_mask.append(eligibility)
                netw_probs.append(probability)
                netw_votes.append(prediction)
            except:
                if verbose:
                    print("Exception in NetworkPrediction for series "+str(series_ids[m]))
                    raise
        if verbose:
            print("IsEligible: "+str(netw_mask))
            print("Netw Probs: "+str(netw_probs))
            print("Netw Votes: "+str(netw_votes))

        # Add network predictions on top of original votes, weighted by the probability assigned by the
        # network, apply voting rules again, then count votes and potentially make a decision
        weights = [prob if prob >= kwargs["netw_conf_threshold"] else 0 for prob in netw_probs]
        nvn = len(netw_votes)
        
        # Reapply voting rules, excluding rtype "!". (This is weaker than the alternative, but with enough
        # series you will eventually mispredict into WB or H or something and then you lose the whole advantage
        # that series voting offered)
        # If we are allowed to use vote_rules on network votes, we combine with the preliminary votes from before
        # and apply the rules to the combination. If not, we stack the network votes on top of the modified
        # votes from before and only do the recount.
        if "vote_rules" in kwargs and "network_vote_rules" in kwargs:
            if kwargs["network_vote_rules"] == True:
                combined_votes = netw_votes + preliminary_votes
                nvn = len(netw_votes)
                if verbose:
                    print("Applying voting rules...")
                for rule in kwargs["vote_rules"]:
                    if rule.rtype != "!":
                        combined_votes, r_is_absolute, r_target = rule.apply_to(combined_votes)
                if verbose:
                    print("=Corrected votes=")
                    kc = 0
                    for key in sorted(list(meta_dict.keys())):
                        if not key in ["Procedure Code", "Study Description", "Series Modality"]:
                            print(str(key)+" Votes:")
                            print(combined_votes[nvn+kc::(len(meta_dict)-3)])
                            kc += 1
                    print("Network Votes:")
                    print(combined_votes[0:nvn])
            else:
                combined_votes = netw_votes + combined_votes
        else:
            combined_votes = netw_votes + combined_votes
        
        # Count all votes again (and count kc if that was not done before)
        kc = 0
        for key in sorted(list(meta_dict.keys())):
            if not key in ["Procedure Code", "Study Description", "Series Modality"]:
                kc += 1
        try:
            vote_hist = {}
            for i, vote in enumerate(combined_votes[nvn:]):
                if mask[int(np.floor(i/(len(meta_dict)-3)))] == 1 and vote:
                    if vote in list(vote_hist.keys()):
                        vote_hist[vote] += 1
                    else:
                        vote_hist[vote] = 1
            for j, vote in enumerate(combined_votes[0:nvn]):
                if netw_mask[j] == 1 and vote:
                    if vote in list(vote_hist.keys()):
                        vote_hist[vote] += weights[j]
                    else:
                        vote_hist[vote] = weights[j]
            winners = []
            most_votes = max(vote_hist.values())
            winners = [key for i, key in enumerate(list(vote_hist.keys())) if list(vote_hist.values())[i] == most_votes]
            if verbose:
                print(vote_hist, most_votes, winners)
        except:
            raise
            if verbose:
                print("Cannot count network series votes.")
            prediction = "UNKNOWN-"+str(STModa)
            decider = "Network failed to make a prediction due to an exception."
            return prediction, decider
        if len(winners) == 0:
            if verbose:
                print("No votes were cast. No eligible series ("+str(valid_modas)+") found or they were no help during substring matching.")
            decider = "Network failed to tiebreak (no votes cast or threshold not met), no prediction will be made."
            prediction = "UNKNOWN-"+str(STModa)
            return prediction, decider
        if len(winners) == 1:
            prediction = winners[0]
            return prediction, decider
        if len(winners) > 1:
            if verbose:
                print("Multiple predictions have same probability.")
            decider = "Network failed to tiebreak (no votes cast or threshold not met), no prediction will be made."
            prediction = "UNKNOWN-"+str(STModa)
            return prediction, decider

def PredictStudy_0(meta_dict, mapfile, network, verbose=False, local=True, split_mode=False, **kwargs):
    """
    As PredictStudy_5, except the NN is the only thing that is used (apart from voting rules making the
    various series predictions into one combined study prediction via the usual vote).
    """
    
    decider = "Neural Network"
    
    # Unpack critical metadata
    try:
        Code = meta_dict["Procedure Code"][0].value
        STDesc = meta_dict["Study Description"][0].value
        SEModas = [item for item in meta_dict["Series Modality"]]
    except KeyError as k:
        print("meta_dict is missing critical metadata key: "+str(k)+". Study will not be predicted.")
        print("If this error shows up constantly, the key was probably excluded in the config.ini.")
        print("If this is the case, consider restoring it.")
        raise
        
    # Valid modalities
    with open(mapfile) as json_file:
        mapping = js.load(json_file)
    valid_modas = []
    for moda in list(mapping["Internal"]["Moda"].values()):
        if moda not in valid_modas and moda==moda:
            valid_modas.append(moda)
            
    # Remapped modalities (pretend all series of type X are actually type Y, generally only for debugging)
    if "remapped_modalities" in kwargs:
        if verbose:
            print("Remapping modalities ...")
        SEModas = [SEM if SEM.value not in kwargs["remapped_modalities"] else meta("Series Modality", kwargs["remapped_modalities"][str(SEM.value)]) for SEM in SEModas]
        
    # Rule-based approach to Modality of Study
    if any(SEModas):
        try:
            tmp = [m.value for m in SEModas if m.value in valid_modas]
            hist = (Counter(tmp))
            STModa = max(hist, key=hist.get)
        except:
            STModa = ""
    else:
        STModa = ""
    if any(x for x in SEModas if x.value == "PT"):
        if STModa == "CT":
            STModa = "PT"
        elif STModa == "MR":
            STModa = "MRPET"
    
    # Classify study as unknown if the modality is entirely unmapped
    if not STModa:
        try:
            tmp = [m.value for m in SEModas if m.value]
            hist = (Counter(tmp))
            STModa = max(hist, key=hist.get)
        except:
            pass
        if not (STModa in valid_modas):
            prediction = "UNKNOWN-"+str(STModa)
            decider = "Unmapped modality provided"
            return prediction, []
        else:
            prediction = "UNKNOWN"
            decider = "No modality provided"
            return prediction, []
    
    # Classify study as MG if STModa == MG because there is only one thing it can be
    if STModa == "MG":
        return "MAM", "Only one solution possible for modality 'MG'."
    
    # Throw it all into the network as per usual
    if verbose:
        print("Feeding data into nothing but the network.")

    # If predictions are made locally, the files need not be downloaded - non-local files must be acquired
    # first, else the network has nothing to evaluate on
    if not local:
        if verbose:
            print("Acquiring data ...")
        # Remove potential leftovers
        DLPath = './tempeval/'
        shutil.rmtree(DLPath)
        os.makedirs(DLPath, exist_ok=True)
        # Download files
        for k in range(len(kwargs["NSeriesIDs"])):
            try:
                Download(NStudyID = kwargs["NStudyID"],
                         NSeriesID = kwargs["NSeriesIDs"][k],
                         dest = DLPath,
                         mode = "all")
            except Exception as e:
                if verbose:
                    print("Download failed. Errmsg: "+str(repr(e)))

        _, file_names, series_ids = GatherSeriesMetadataFromStudy(data_root = str(DLPath),
                                                                  known_metas = kwargs["known_metas"],
                                                                  verbose = verbose)
    else:
        file_names = kwargs["file_names"]
        series_ids = kwargs["series_ids"]

    # If split mode, figure out network/mapfile, then predict, else predict with default network/mapfile
    if split_mode:
        mapfile = kwargs["mapfiles"][str(STModa)]
        network = kwargs["networks"][str(STModa)]

    netw_mask = []
    netw_probs = []
    netw_votes = []

    for m in range(len(series_ids)):
        try:
            # Default series descriptions for emergencies
            if not "Series Description" in list(meta_dict.keys()):
                meta_dict["Series Description"] = []
                for i in range(len(meta_dict["Series Modality"])):
                    meta_dict["Series Description"].append(meta("Series Description", "None"))
            eligibility, probability, prediction = PredictSeriesWithNetwork(STModa = STModa,
                                                                            SEDesc = meta_dict["Series Description"][m].value,
                                                                            SEModa = SEModas[m].value,
                                                                            SEFN = file_names[m],
                                                                            SEID = series_ids[m],
                                                                            mapfile = mapfile,
                                                                            network = network,
                                                                            verbose = verbose,
                                                                            **kwargs)
            netw_mask.append(eligibility)
            netw_probs.append(probability)
            netw_votes.append(prediction)
        except:
            if verbose:
                print("Exception in NetworkPrediction for series "+str(series_ids[m]))
                raise
    if verbose:
        print("IsEligible: "+str(netw_mask))
        print("Netw Probs: "+str(netw_probs))
        print("Netw Votes: "+str(netw_votes))

    weights = [prob if prob >= kwargs["netw_conf_threshold"] else 0 for prob in netw_probs]

    # Apply voting rules that are not rtype "!". (This is weaker than the alternative, but with enough
    # series you will eventually mispredict into WB or H or something and then you lose the whole advantage
    # that series voting offered, that's sadly just statistics)
    corr_votes = netw_votes
    if "vote_rules" in kwargs and "network_vote_rules" in kwargs:
        if kwargs["network_vote_rules"] == True:
            if verbose:
                print("Applying voting rules...")
            for rule in kwargs["vote_rules"]:
                if rule.rtype != "!":
                    corr_votes, r_is_absolute, r_target = rule.apply_to(corr_votes)
            if verbose:
                print("=Corrected votes=")
                print("Network Votes:")
                print(corr_votes)

    # Count the votes
    try:
        vote_hist = {}
        for j, vote in enumerate(corr_votes):
            if netw_mask[j] == 1 and vote:
                if vote in list(vote_hist.keys()):
                    vote_hist[vote] += weights[j]
                else:
                    vote_hist[vote] = weights[j]
        winners = []
        most_votes = max(vote_hist.values())
        winners = [key for i, key in enumerate(list(vote_hist.keys())) if list(vote_hist.values())[i] == most_votes]
        if verbose:
            print(vote_hist, most_votes, winners)
    except:
        if verbose:
            print("Cannot count network series votes.")
        prediction = "UNKNOWN-"+str(STModa)
        decider = "Network failed to make a prediction due to an exception."
        return prediction, []
    if len(winners) == 0:
        if verbose:
            print("No votes were cast. No eligible series ("+str(valid_modas)+") found or they were no help during substring matching.")
        decider = "Network failed to tiebreak (no votes cast or threshold not met), no prediction will be made."
        prediction = "UNKNOWN-"+str(STModa)
        return prediction, []
    if len(winners) == 1:
        decider = "Neural network"
        prediction = winners[0]
        return prediction, netw_votes
    if len(winners) >1:
        if verbose:
            print("Multiple predictions have same probability. No decision is made.")
        decider = "Network failed to tiebreak (no votes cast or threshold not met), no prediction will be made."
        prediction = "UNKNOWN-"+str(STModa)
        return prediction, []

def PredictStudy_2(meta_dict, mapfile, network, verbose=False, local=True, split_mode=False, **kwargs):
    """
    As PredictStudy_5, except the NN is swapped to layer 2.
    """
    
    # Unpack critical metadata
    try:
        Code = meta_dict["Procedure Code"][0].value
        STDesc = meta_dict["Study Description"][0].value
        SEModas = [item for item in meta_dict["Series Modality"]]
    except KeyError as k:
        print("meta_dict is missing critical metadata key: "+str(k)+". Study will not be predicted.")
        print("If this error shows up constantly, the key was probably excluded in the config.ini.")
        print("If this is the case, consider restoring it.")
        raise
        
    # Valid modalities
    with open(mapfile) as json_file:
        mapping = js.load(json_file)
    valid_modas = []
    for moda in list(mapping["Internal"]["Moda"].values()):
        if moda not in valid_modas and moda==moda:
            valid_modas.append(moda)
            
    # Remapped modalities (pretend all series of type X are actually type Y, generally only for debugging)
    if "remapped_modalities" in kwargs:
        if verbose:
            print("Remapping modalities ...")
        SEModas = [SEM if SEM.value not in kwargs["remapped_modalities"] else meta("Series Modality", kwargs["remapped_modalities"][str(SEM.value)]) for SEM in SEModas]

    # Rule-based approach to Modality of Study
    if any(SEModas):
        try:
            tmp = [m.value for m in SEModas if m.value in valid_modas]
            hist = (Counter(tmp))
            STModa = max(hist, key=hist.get)
        except:
            STModa = ""
    else:
        STModa = ""
    if any(x for x in SEModas if x.value == "PT"):
        if STModa == "CT":
            STModa = "PT"
        elif STModa == "MR":
            STModa = "MRPET"
            
    # Classify study as unknown if the modality is entirely unmapped
    if not STModa:
        try:
            tmp = [m.value for m in SEModas if m.value]
            hist = (Counter(tmp))
            STModa = max(hist, key=hist.get)
        except:
            pass
        if not (STModa in valid_modas):
            prediction = "UNKNOWN-"+str(STModa)
            decider = "Unmapped modality provided"
            return prediction, decider
        else:
            prediction = "UNKNOWN"
            decider = "No modality provided"
            return prediction, decider
    
    ####################################################################################################
    # Layer 1
    ####################################################################################################
    if verbose:
        print("Trying to match procedure code.")
    decider = "Code, Full Match"
    try:
        c = list(mapping["Internal"]["Code"].keys())[list(mapping["Internal"]["Code"].values()).index(str(Code))]
        prediction = mapping["Internal"]["Code"][str(c)]
        if verbose:
            print("Full match for procedure code found.")
        return prediction, decider
    except:
        # No match for "Code", try Multiclass
        try:
            prediction = mapping["Internal"]["Multiclass"][str(Code)]
            if verbose:
                print("Full match for procedure code found.")
            return prediction, decider
        except:
            # No matches
            if verbose:
                print("No exact match for Procedure Code in mapping.")
                
    ####################################################################################################
    # Layer 2 (Network)
    ####################################################################################################
    # Classify study as MG if STModa == MG because there is only one thing it can be
    if STModa == "MG":
        return "MAM", "Only one solution possible for modality 'MG'."
    
    # If predictions are made locally, the files need not be downloaded - non-local files must be acquired
    # first, else the network has nothing to evaluate on
    if not local:
        if verbose:
            print("Acquiring data ...")
        # Remove potential leftovers
        DLPath = './tempeval/'
        shutil.rmtree(DLPath)
        os.makedirs(DLPath, exist_ok=True)
        # Download files
        for k in range(len(kwargs["NSeriesIDs"])):
            try:
                Download(NStudyID = kwargs["NStudyID"],
                         NSeriesID = kwargs["NSeriesIDs"][k],
                         dest = DLPath,
                         mode = "all")
            except Exception as e:
                if verbose:
                    print("Download failed. Errmsg: "+str(repr(e)))

        _, file_names, series_ids = GatherSeriesMetadataFromStudy(data_root = str(DLPath),
                                                                  known_metas = kwargs["known_metas"],
                                                                  verbose = verbose)
    else:
        file_names = kwargs["file_names"]
        series_ids = kwargs["series_ids"]

    # If split mode, figure out network/mapfile, then predict, else predict with default network/mapfile
    if split_mode:
        mapfile = kwargs["mapfiles"][str(STModa)]
        network = kwargs["networks"][str(STModa)]

    netw_mask = []
    netw_probs = []
    netw_votes = []

    for m in range(len(series_ids)):
        try:
            # Default series descriptions for emergencies
            if not "Series Description" in list(meta_dict.keys()):
                meta_dict["Series Description"] = []
                for i in range(len(meta_dict["Series Modality"])):
                    meta_dict["Series Description"].append(meta("Series Description", "None"))
            eligibility, probability, prediction = PredictSeriesWithNetwork(STModa = STModa,
                                                                            SEDesc = meta_dict["Series Description"][m].value,
                                                                            SEModa = SEModas[m].value,
                                                                            SEFN = file_names[m],
                                                                            SEID = series_ids[m],
                                                                            mapfile = mapfile,
                                                                            network = network,
                                                                            verbose = verbose,
                                                                            **kwargs)
            netw_mask.append(eligibility)
            netw_probs.append(probability)
            netw_votes.append(prediction)
        except:
            if verbose:
                print("Exception in NetworkPrediction for series "+str(series_ids[m]))
                raise
    if verbose:
        print("IsEligible: "+str(netw_mask))
        print("Netw Probs: "+str(netw_probs))
        print("Netw Votes: "+str(netw_votes))

    weights = [prob if prob >= kwargs["netw_conf_threshold"] else 0 for prob in netw_probs]

    # Apply voting rules that are not rtype "!". (This is weaker than the alternative, but with enough
    # series you will eventually mispredict into WB or H or something and then you lose the whole advantage
    # that series voting offered, that's sadly just statistics)
    corr_votes = netw_votes
    if "vote_rules" in kwargs and "network_vote_rules" in kwargs:
        if kwargs["network_vote_rules"] == True:
            if verbose:
                print("Applying voting rules...")
            for rule in kwargs["vote_rules"]:
                if rule.rtype != "!":
                    corr_votes, r_is_absolute, r_target = rule.apply_to(corr_votes)
            if verbose:
                print("=Corrected votes=")
                print("Network Votes:")
                print(corr_votes)

    # Count the votes
    try:
        vote_hist = {}
        for j, vote in enumerate(corr_votes):
            if netw_mask[j] == 1 and vote:
                if vote in list(vote_hist.keys()):
                    vote_hist[vote] += weights[j]
                else:
                    vote_hist[vote] = weights[j]
        winners = []
        most_votes = max(vote_hist.values())
        winners = [key for i, key in enumerate(list(vote_hist.keys())) if list(vote_hist.values())[i] == most_votes]
        if verbose:
            print(vote_hist, most_votes, winners)
    except:
        if verbose:
            print("Cannot count network series votes.")
        prediction = "UNKNOWN-"+str(STModa)
        decider = "Network failed to make a prediction due to an exception."
    if len(winners) == 0:
        if verbose:
            print("No votes were cast. No eligible series ("+str(valid_modas)+") found or they were no help during substring matching.")
        decider = "Network failed to tiebreak (no votes cast or threshold not met), no prediction will be made."
        prediction = "UNKNOWN-"+str(STModa)
    if len(winners) == 1:
        decider = "Neural network"
        prediction = winners[0]
        return prediction, decider
    if len(winners) >1:
        if verbose:
            print("Multiple predictions have same probability. No decision is made.")
        decider = "Network failed to tiebreak (no votes cast or threshold not met), no prediction will be made."
        prediction = "UNKNOWN-"+str(STModa)
        
    ####################################################################################################
    # Layer 3
    ####################################################################################################
    if verbose:
        print("Trying to match study description.")
    decider = "Reduced Study Description, Full Match"
    try:
        c = Reduce(list(mapping["Internal"]["Desc"].keys()))[Reduce(list(mapping["Internal"]["Desc"].values())).index(str(Reduce(STDesc)))]
        prediction = mapping["Internal"]["Code"][str(c)]
        # Only accept full match if modality is correct (unless no modality is given, in which case, accept all)
        if mapping["Internal"]["Moda"][str(c)] == STModa or not STModa:
            if verbose:
                print("Full match for Study Description found.")
            return prediction, decider
        else:
            if verbose:
                print("Exact match for study description in mapping, but wrong modality.")
    except:
        # No exact match
        if verbose:
            print("No exact match for study description in mapping.")
    ####################################################################################################
    # Layer 4
    ####################################################################################################
    if verbose:
        print("Trying to match substrings from collected metadata.")
    decider = "Voting (1 vote each best substring match over all collected metadata)"
    
    # Grab all descriptions from mapping which are legal, based on known modality
    # If modality is unknown or not one of the main ones, allow all modalities
    # Grab all descriptions from mapping which are legal, based on known modality
    if STModa == "MRPET":
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
    elif STModa == "MR":
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
    elif not STModa:
        # Making sure to exclude the nan variant from the JSON
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if key==key]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if val==val]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if key==key]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if val==val]
    else:
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
    
    # If there is only one item in keys, we do not need to match anything, this is the only possibility
    # Typically, this should only ever apply to MG, but who knows. If this is incorrect, then the mapping
    # needs to be adjusted.
    desclen = len(keys)/2
    if desclen == 1:
        stdesc_prediction = mapping["Internal"]["Code"][str(keys[0])]
        if verbose:
            print("Modality has only one entry in mapping, this should be the correct class.")
        return stdesc_prediction, decider
    
    # Add the modality to all keywords, and once at the end for description, add PT as PETCT
    if STModa == "PT":
        ML = ["PT", "PET CT", "CT PET"]
    elif STModa == "MR":
        ML = ["MRT", "MR"]
    elif STModa == "MRPET":
        ML = ["MR PET", "PET MR", "MRT PET", "PET MRT"]
    else:
        ML = [STModa]
    for idx, alts in enumerate(vals):
        if idx < desclen:
            vals[idx] = str(";")+";".join([vals[idx]+str(";")+vals[idx]+M for M in ML])+str(";")
        else:
            vals[idx] = str(";")+";".join(";".join([item+M+str(";")+M+item+str(";")+item for item in alts.split(";")]) for M in ML)+str(";")

    # Substring matching
    if "minmatchlength" in kwargs:
        meml = kwargs["minmatchlength"]["study"][0]
        mrml = kwargs["minmatchlength"]["study"][0]
    else:
        meml = 4
        mrml = 6
    stdesc_prediction, is_exact = SubstringMatcher(keys = keys,
                                                   vals = Reduce(vals),
                                                   desc = STDesc, 
                                                   meml = meml, 
                                                   mrml = mrml, 
                                                   mapping = mapping, 
                                                   verbose = verbose, 
                                                   return_exact = True, 
                                                   **kwargs)
            
    # Instant hits - If we score an exact match in the study description, there is no reason to continue and
    # we immediately return the value
    if is_exact:
        decider = "Best substring match for Study Description"
        if verbose:
            print("Study description had exact match, this is our best guess.")
        return stdesc_prediction, decider
    
    ####################################################################################################
    # Layer 5
    ####################################################################################################
    # Series Voting starts here, try matching as above and then vote by simple majority for a prediction
    if verbose:
        print("Beginning series voting...")
    decider = "Series Voting"

    preliminary_votes = []
    mask = []
    for i in range(len(meta_dict["Procedure Code"])):
        eligibility, votes = PredictSeries(meta_dict = meta_dict,
                                           sidx = i,
                                           STModa = STModa,
                                           mapping = mapping,
                                           verbose = verbose,
                                           **kwargs)
        preliminary_votes.extend(votes)
        mask.append(eligibility)
    
    if verbose:
        print("=Preliminary votes=")
        kc = 0
        for key in sorted(list(meta_dict.keys())):
            if not key in ["Procedure Code", "Study Description", "Series Modality"]:
                print(str(key)+" Votes:")
                print(preliminary_votes[kc::(len(meta_dict)-3)])
                kc += 1
        print("Is eligible:")
        print(mask)
        
    # Apply voting rules (if a rule establishes an absolute, exit rules application. (All legal votes have been set
    # to that absolute value anyway, meaning we can just count votes as we normally would and the absolute wins.)
    if "vote_rules" in kwargs:
        if verbose:
            print("Applying voting rules...")
        combined_votes = preliminary_votes
        for rule in kwargs["vote_rules"]:
            combined_votes, r_is_absolute, r_target = rule.apply_to(combined_votes)
            if r_is_absolute:
                break
        if verbose:
            print("=Corrected votes=")
            kc = 0
            for key in sorted(list(meta_dict.keys())):
                if not key in ["Procedure Code", "Study Description", "Series Modality"]:
                    print(str(key)+" Votes:")
                    print(combined_votes[kc::(len(meta_dict)-3)])
                    kc += 1
            print("Is eligible:")
            print(mask)
    
    # Count all votes, if the vote is not None and the voting series was deemed eligible
    winners = []
    try:
        vote_hist = Counter([vote for i,vote in enumerate(combined_votes) if (mask[int(np.floor(i/(len(meta_dict)-3)))] == 1 and vote)])
        most_votes = max(vote_hist.values())
        winners = [key for i, key in enumerate(list(vote_hist.keys())) if list(vote_hist.values())[i] == most_votes]
        if verbose:
            print(vote_hist, most_votes, winners)
    except:
        if verbose:
            print("Cannot count series votes.")
    if len(winners) == 0:
        if verbose:
            print("No votes were cast. No eligible series ("+str(valid_modas)+") found or they were no help during substring matching.")
        prediction = "UNKNOWN-"+str(STModa)
        decider = "Series vote could not make a prediction."
        return prediction, decider
    if len(winners) == 1:
        return winners[0], decider
    if len(winners) > 1:
        prediction = "UNKNOWN-"+str(STModa)
        decider = "Series vote predicted multiple classes with same number of votes."
        return prediction, decider
    
def PredictStudy_3(meta_dict, mapfile, network, verbose=False, local=True, split_mode=False, **kwargs):
    """
    As PredictStudy_5, except the NN is swapped to layer 3.
    """
    
    # Unpack critical metadata
    try:
        Code = meta_dict["Procedure Code"][0].value
        STDesc = meta_dict["Study Description"][0].value
        SEModas = [item for item in meta_dict["Series Modality"]]
    except KeyError as k:
        print("meta_dict is missing critical metadata key: "+str(k)+". Study will not be predicted.")
        print("If this error shows up constantly, the key was probably excluded in the config.ini.")
        print("If this is the case, consider restoring it.")
        raise
        
    # Valid modalities
    with open(mapfile) as json_file:
        mapping = js.load(json_file)
    valid_modas = []
    for moda in list(mapping["Internal"]["Moda"].values()):
        if moda not in valid_modas and moda==moda:
            valid_modas.append(moda)
            
    # Remapped modalities (pretend all series of type X are actually type Y, generally only for debugging)
    if "remapped_modalities" in kwargs:
        if verbose:
            print("Remapping modalities ...")
        SEModas = [SEM if SEM.value not in kwargs["remapped_modalities"] else meta("Series Modality", kwargs["remapped_modalities"][str(SEM.value)]) for SEM in SEModas]

    # Rule-based approach to Modality of Study
    if any(SEModas):
        try:
            tmp = [m.value for m in SEModas if m.value in valid_modas]
            hist = (Counter(tmp))
            STModa = max(hist, key=hist.get)
        except:
            STModa = ""
    else:
        STModa = ""
    if any(x for x in SEModas if x.value == "PT"):
        if STModa == "CT":
            STModa = "PT"
        elif STModa == "MR":
            STModa = "MRPET"
            
    # Classify study as unknown if the modality is entirely unmapped
    if not STModa:
        try:
            tmp = [m.value for m in SEModas if m.value]
            hist = (Counter(tmp))
            STModa = max(hist, key=hist.get)
        except:
            pass
        if not (STModa in valid_modas):
            prediction = "UNKNOWN-"+str(STModa)
            decider = "Unmapped modality provided"
            return prediction, decider
        else:
            prediction = "UNKNOWN"
            decider = "No modality provided"
            return prediction, decider
    
    ####################################################################################################
    # Layer 1
    ####################################################################################################
    if verbose:
        print("Trying to match procedure code.")
    decider = "Code, Full Match"
    try:
        c = list(mapping["Internal"]["Code"].keys())[list(mapping["Internal"]["Code"].values()).index(str(Code))]
        prediction = mapping["Internal"]["Code"][str(c)]
        if verbose:
            print("Full match for procedure code found.")
        return prediction, decider
    except:
        # No match for "Code", try Multiclass
        try:
            prediction = mapping["Internal"]["Multiclass"][str(Code)]
            if verbose:
                print("Full match for procedure code found.")
            return prediction, decider
        except:
            # No matches
            if verbose:
                print("No exact match for Procedure Code in mapping.")
        
    ####################################################################################################
    # Layer 2
    ####################################################################################################
    if verbose:
        print("Trying to match study description.")
    decider = "Reduced Study Description, Full Match"
    try:
        c = Reduce(list(mapping["Internal"]["Desc"].keys()))[Reduce(list(mapping["Internal"]["Desc"].values())).index(str(Reduce(STDesc)))]
        prediction = mapping["Internal"]["Code"][str(c)]
        # Only accept full match if modality is correct (unless no modality is given, in which case, accept all)
        if mapping["Internal"]["Moda"][str(c)] == STModa or not STModa:
            if verbose:
                print("Full match for Study Description found.")
            return prediction, decider
        else:
            if verbose:
                print("Exact match for study description in mapping, but wrong modality.")
    except:
        # No exact match
        if verbose:
            print("No exact match for study description in mapping.")
            
    ####################################################################################################
    # Layer 3 (Network)
    ####################################################################################################
    # Classify study as MG if STModa == MG because there is only one thing it can be
    if STModa == "MG":
        return "MAM", "Only one solution possible for modality 'MG'."
    
    # If predictions are made locally, the files need not be downloaded - non-local files must be acquired
    # first, else the network has nothing to evaluate on
    if not local:
        if verbose:
            print("Acquiring data ...")
        # Remove potential leftovers
        DLPath = './tempeval/'
        shutil.rmtree(DLPath)
        os.makedirs(DLPath, exist_ok=True)
        # Download files
        for k in range(len(kwargs["NSeriesIDs"])):
            try:
                Download(NStudyID = kwargs["NStudyID"],
                         NSeriesID = kwargs["NSeriesIDs"][k],
                         dest = DLPath,
                         mode = "all")
            except Exception as e:
                if verbose:
                    print("Download failed. Errmsg: "+str(repr(e)))

        _, file_names, series_ids = GatherSeriesMetadataFromStudy(data_root = str(DLPath),
                                                                  known_metas = kwargs["known_metas"],
                                                                  verbose = verbose)
    else:
        file_names = kwargs["file_names"]
        series_ids = kwargs["series_ids"]

    # If split mode, figure out network/mapfile, then predict, else predict with default network/mapfile
    if split_mode:
        mapfile = kwargs["mapfiles"][str(STModa)]
        network = kwargs["networks"][str(STModa)]

    netw_mask = []
    netw_probs = []
    netw_votes = []

    for m in range(len(series_ids)):
        try:
            # Default series descriptions for emergencies
            if not "Series Description" in list(meta_dict.keys()):
                meta_dict["Series Description"] = []
                for i in range(len(meta_dict["Series Modality"])):
                    meta_dict["Series Description"].append(meta("Series Description", "None"))
            eligibility, probability, prediction = PredictSeriesWithNetwork(STModa = STModa,
                                                                            SEDesc = meta_dict["Series Description"][m].value,
                                                                            SEModa = SEModas[m].value,
                                                                            SEFN = file_names[m],
                                                                            SEID = series_ids[m],
                                                                            mapfile = mapfile,
                                                                            network = network,
                                                                            verbose = verbose,
                                                                            **kwargs)
            netw_mask.append(eligibility)
            netw_probs.append(probability)
            netw_votes.append(prediction)
        except:
            if verbose:
                print("Exception in NetworkPrediction for series "+str(series_ids[m]))
                raise
    if verbose:
        print("IsEligible: "+str(netw_mask))
        print("Netw Probs: "+str(netw_probs))
        print("Netw Votes: "+str(netw_votes))

    weights = [prob if prob >= kwargs["netw_conf_threshold"] else 0 for prob in netw_probs]

    # Apply voting rules that are not rtype "!". (This is weaker than the alternative, but with enough
    # series you will eventually mispredict into WB or H or something and then you lose the whole advantage
    # that series voting offered, that's sadly just statistics)
    corr_votes = netw_votes
    if "vote_rules" in kwargs and "network_vote_rules" in kwargs:
        if kwargs["network_vote_rules"] == True:
            if verbose:
                print("Applying voting rules...")
            for rule in kwargs["vote_rules"]:
                if rule.rtype != "!":
                    corr_votes, r_is_absolute, r_target = rule.apply_to(corr_votes)
            if verbose:
                print("=Corrected votes=")
                print("Network Votes:")
                print(corr_votes)

    # Count the votes
    try:
        vote_hist = {}
        for j, vote in enumerate(corr_votes):
            if netw_mask[j] == 1 and vote:
                if vote in list(vote_hist.keys()):
                    vote_hist[vote] += weights[j]
                else:
                    vote_hist[vote] = weights[j]
        winners = []
        most_votes = max(vote_hist.values())
        winners = [key for i, key in enumerate(list(vote_hist.keys())) if list(vote_hist.values())[i] == most_votes]
        if verbose:
            print(vote_hist, most_votes, winners)
    except:
        if verbose:
            print("Cannot count network series votes.")
        prediction = "UNKNOWN-"+str(STModa)
        decider = "Network failed to make a prediction due to an exception."
    if len(winners) == 0:
        if verbose:
            print("No votes were cast. No eligible series ("+str(valid_modas)+") found or they were no help during substring matching.")
        decider = "Network failed to tiebreak (no votes cast or threshold not met), no prediction will be made."
        prediction = "UNKNOWN-"+str(STModa)
    if len(winners) == 1:
        decider = "Neural network"
        prediction = winners[0]
        return prediction, decider
    if len(winners) >1:
        if verbose:
            print("Multiple predictions have same probability. No decision is made.")
        decider = "Network failed to tiebreak (no votes cast or threshold not met), no prediction will be made."
        prediction = "UNKNOWN-"+str(STModa)
            
    ####################################################################################################
    # Layer 4
    ####################################################################################################
    if verbose:
        print("Trying to match substrings from collected metadata.")
    decider = "Voting (1 vote each best substring match over all collected metadata)"
    
    # Grab all descriptions from mapping which are legal, based on known modality
    # If modality is unknown or not one of the main ones, allow all modalities
    # Grab all descriptions from mapping which are legal, based on known modality
    if STModa == "MRPET":
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
    elif STModa == "MR":
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
    elif not STModa:
        # Making sure to exclude the nan variant from the JSON
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if key==key]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if val==val]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if key==key]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if val==val]
    else:
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
    
    # If there is only one item in keys, we do not need to match anything, this is the only possibility
    # Typically, this should only ever apply to MG, but who knows. If this is incorrect, then the mapping
    # needs to be adjusted.
    desclen = len(keys)/2
    if desclen == 1:
        stdesc_prediction = mapping["Internal"]["Code"][str(keys[0])]
        if verbose:
            print("Modality has only one entry in mapping, this should be the correct class.")
        return stdesc_prediction, decider
    
    # Add the modality to all keywords, and once at the end for description, add PT as PETCT
    if STModa == "PT":
        ML = ["PT", "PET CT", "CT PET"]
    elif STModa == "MR":
        ML = ["MRT", "MR"]
    elif STModa == "MRPET":
        ML = ["MR PET", "PET MR", "MRT PET", "PET MRT"]
    else:
        ML = [STModa]
    for idx, alts in enumerate(vals):
        if idx < desclen:
            vals[idx] = str(";")+";".join([vals[idx]+str(";")+vals[idx]+M for M in ML])+str(";")
        else:
            vals[idx] = str(";")+";".join(";".join([item+M+str(";")+M+item+str(";")+item for item in alts.split(";")]) for M in ML)+str(";")

    # Substring matching
    if "minmatchlength" in kwargs:
        meml = kwargs["minmatchlength"]["study"][0]
        mrml = kwargs["minmatchlength"]["study"][0]
    else:
        meml = 4
        mrml = 6
    stdesc_prediction, is_exact = SubstringMatcher(keys = keys,
                                                   vals = Reduce(vals),
                                                   desc = STDesc, 
                                                   meml = meml, 
                                                   mrml = mrml, 
                                                   mapping = mapping, 
                                                   verbose = verbose, 
                                                   return_exact = True, 
                                                   **kwargs)
            
    # Instant hits - If we score an exact match in the study description, there is no reason to continue and
    # we immediately return the value
    if is_exact:
        decider = "Best substring match for Study Description"
        if verbose:
            print("Study description had exact match, this is our best guess.")
        return stdesc_prediction, decider
    
    ####################################################################################################
    # Layer 5
    ####################################################################################################
    # Series Voting starts here, try matching as above and then vote by simple majority for a prediction
    if verbose:
        print("Beginning series voting...")
    decider = "Series Voting"

    preliminary_votes = []
    mask = []
    for i in range(len(meta_dict["Procedure Code"])):
        eligibility, votes = PredictSeries(meta_dict = meta_dict,
                                           sidx = i,
                                           STModa = STModa,
                                           mapping = mapping,
                                           verbose = verbose,
                                           **kwargs)
        preliminary_votes.extend(votes)
        mask.append(eligibility)
    
    if verbose:
        print("=Preliminary votes=")
        kc = 0
        for key in sorted(list(meta_dict.keys())):
            if not key in ["Procedure Code", "Study Description", "Series Modality"]:
                print(str(key)+" Votes:")
                print(preliminary_votes[kc::(len(meta_dict)-3)])
                kc += 1
        print("Is eligible:")
        print(mask)
        
    # Apply voting rules (if a rule establishes an absolute, exit rules application. (All legal votes have been set
    # to that absolute value anyway, meaning we can just count votes as we normally would and the absolute wins.)
    if "vote_rules" in kwargs:
        if verbose:
            print("Applying voting rules...")
        combined_votes = preliminary_votes
        for rule in kwargs["vote_rules"]:
            combined_votes, r_is_absolute, r_target = rule.apply_to(combined_votes)
            if r_is_absolute:
                break
        if verbose:
            print("=Corrected votes=")
            kc = 0
            for key in sorted(list(meta_dict.keys())):
                if not key in ["Procedure Code", "Study Description", "Series Modality"]:
                    print(str(key)+" Votes:")
                    print(combined_votes[kc::(len(meta_dict)-3)])
                    kc += 1
            print("Is eligible:")
            print(mask)
    
    # Count all votes, if the vote is not None and the voting series was deemed eligible
    winners = []
    try:
        vote_hist = Counter([vote for i,vote in enumerate(combined_votes) if (mask[int(np.floor(i/(len(meta_dict)-3)))] == 1 and vote)])
        most_votes = max(vote_hist.values())
        winners = [key for i, key in enumerate(list(vote_hist.keys())) if list(vote_hist.values())[i] == most_votes]
        if verbose:
            print(vote_hist, most_votes, winners)
    except:
        if verbose:
            print("Cannot count series votes.")
    if len(winners) == 0:
        if verbose:
            print("No votes were cast. No eligible series ("+str(valid_modas)+") found or they were no help during substring matching.")
        prediction = "UNKNOWN-"+str(STModa)
        decider = "Series vote could not make a prediction."
        return prediction, decider
    if len(winners) == 1:
        return winners[0], decider
    if len(winners) > 1:
        prediction = "UNKNOWN-"+str(STModa)
        decider = "Series vote predicted multiple classes with same number of votes."
        return prediction, decider
    
def PredictStudy_4(meta_dict, mapfile, network, verbose=False, local=True, split_mode=False, **kwargs):
    """
    As PredictStudy_5, except the NN is swapped to layer 4.
    """
    
    # Unpack critical metadata
    try:
        Code = meta_dict["Procedure Code"][0].value
        STDesc = meta_dict["Study Description"][0].value
        SEModas = [item for item in meta_dict["Series Modality"]]
    except KeyError as k:
        print("meta_dict is missing critical metadata key: "+str(k)+". Study will not be predicted.")
        print("If this error shows up constantly, the key was probably excluded in the config.ini.")
        print("If this is the case, consider restoring it.")
        raise
        
    # Valid modalities
    with open(mapfile) as json_file:
        mapping = js.load(json_file)
    valid_modas = []
    for moda in list(mapping["Internal"]["Moda"].values()):
        if moda not in valid_modas and moda==moda:
            valid_modas.append(moda)
            
    # Remapped modalities (pretend all series of type X are actually type Y, generally only for debugging)
    if "remapped_modalities" in kwargs:
        if verbose:
            print("Remapping modalities ...")
        SEModas = [SEM if SEM.value not in kwargs["remapped_modalities"] else meta("Series Modality", kwargs["remapped_modalities"][str(SEM.value)]) for SEM in SEModas]

    # Rule-based approach to Modality of Study
    if any(SEModas):
        try:
            tmp = [m.value for m in SEModas if m.value in valid_modas]
            hist = (Counter(tmp))
            STModa = max(hist, key=hist.get)
        except:
            STModa = ""
    else:
        STModa = ""
    if any(x for x in SEModas if x.value == "PT"):
        if STModa == "CT":
            STModa = "PT"
        elif STModa == "MR":
            STModa = "MRPET"
            
    # Classify study as unknown if the modality is entirely unmapped
    if not STModa:
        try:
            tmp = [m.value for m in SEModas if m.value]
            hist = (Counter(tmp))
            STModa = max(hist, key=hist.get)
        except:
            pass
        if not (STModa in valid_modas):
            prediction = "UNKNOWN-"+str(STModa)
            decider = "Unmapped modality provided"
            return prediction, decider
        else:
            prediction = "UNKNOWN"
            decider = "No modality provided"
            return prediction, decider
    
    ####################################################################################################
    # Layer 1
    ####################################################################################################
    if verbose:
        print("Trying to match procedure code.")
    decider = "Code, Full Match"
    try:
        c = list(mapping["Internal"]["Code"].keys())[list(mapping["Internal"]["Code"].values()).index(str(Code))]
        prediction = mapping["Internal"]["Code"][str(c)]
        if verbose:
            print("Full match for procedure code found.")
        return prediction, decider
    except:
        # No match for "Code", try Multiclass
        try:
            prediction = mapping["Internal"]["Multiclass"][str(Code)]
            if verbose:
                print("Full match for procedure code found.")
            return prediction, decider
        except:
            # No matches
            if verbose:
                print("No exact match for Procedure Code in mapping.")
        
    ####################################################################################################
    # Layer 2
    ####################################################################################################
    if verbose:
        print("Trying to match study description.")
    decider = "Reduced Study Description, Full Match"
    try:
        c = Reduce(list(mapping["Internal"]["Desc"].keys()))[Reduce(list(mapping["Internal"]["Desc"].values())).index(str(Reduce(STDesc)))]
        prediction = mapping["Internal"]["Code"][str(c)]
        # Only accept full match if modality is correct (unless no modality is given, in which case, accept all)
        if mapping["Internal"]["Moda"][str(c)] == STModa or not STModa:
            if verbose:
                print("Full match for Study Description found.")
            return prediction, decider
        else:
            if verbose:
                print("Exact match for study description in mapping, but wrong modality.")
    except:
        # No exact match
        if verbose:
            print("No exact match for study description in mapping.")
            
    ####################################################################################################
    # Layer 3
    ####################################################################################################
    if verbose:
        print("Trying to match substrings from collected metadata.")
    decider = "Voting (1 vote each best substring match over all collected metadata)"
    
    # Grab all descriptions from mapping which are legal, based on known modality
    # If modality is unknown or not one of the main ones, allow all modalities
    # Grab all descriptions from mapping which are legal, based on known modality
    if STModa == "MRPET":
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
    elif STModa == "MR":
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
    elif not STModa:
        # Making sure to exclude the nan variant from the JSON
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if key==key]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if val==val]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if key==key]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if val==val]
    else:
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
    
    # If there is only one item in keys, we do not need to match anything, this is the only possibility
    # Typically, this should only ever apply to MG, but who knows. If this is incorrect, then the mapping
    # needs to be adjusted.
    desclen = len(keys)/2
    if desclen == 1:
        stdesc_prediction = mapping["Internal"]["Code"][str(keys[0])]
        if verbose:
            print("Modality has only one entry in mapping, this should be the correct class.")
        return stdesc_prediction, decider
    
    # Add the modality to all keywords, and once at the end for description, add PT as PETCT
    if STModa == "PT":
        ML = ["PT", "PET CT", "CT PET"]
    elif STModa == "MR":
        ML = ["MRT", "MR"]
    elif STModa == "MRPET":
        ML = ["MR PET", "PET MR", "MRT PET", "PET MRT"]
    else:
        ML = [STModa]
    for idx, alts in enumerate(vals):
        if idx < desclen:
            vals[idx] = str(";")+";".join([vals[idx]+str(";")+vals[idx]+M for M in ML])+str(";")
        else:
            vals[idx] = str(";")+";".join(";".join([item+M+str(";")+M+item+str(";")+item for item in alts.split(";")]) for M in ML)+str(";")

    # Substring matching
    if "minmatchlength" in kwargs:
        meml = kwargs["minmatchlength"]["study"][0]
        mrml = kwargs["minmatchlength"]["study"][0]
    else:
        meml = 4
        mrml = 6
    stdesc_prediction, is_exact = SubstringMatcher(keys = keys,
                                                   vals = Reduce(vals),
                                                   desc = STDesc, 
                                                   meml = meml, 
                                                   mrml = mrml, 
                                                   mapping = mapping, 
                                                   verbose = verbose, 
                                                   return_exact = True, 
                                                   **kwargs)
            
    # Instant hits - If we score an exact match in the study description, there is no reason to continue and
    # we immediately return the value
    if is_exact:
        decider = "Best substring match for Study Description"
        if verbose:
            print("Study description had exact match, this is our best guess.")
        return stdesc_prediction, decider
    
    ####################################################################################################
    # Layer 4 (Network)
    ####################################################################################################
    # Classify study as MG if STModa == MG because there is only one thing it can be
    if STModa == "MG":
        return "MAM", "Only one solution possible for modality 'MG'."
    
    # If predictions are made locally, the files need not be downloaded - non-local files must be acquired
    # first, else the network has nothing to evaluate on
    if not local:
        if verbose:
            print("Acquiring data ...")
        # Remove potential leftovers
        DLPath = './tempeval/'
        shutil.rmtree(DLPath)
        os.makedirs(DLPath, exist_ok=True)
        # Download files
        for k in range(len(kwargs["NSeriesIDs"])):
            try:
                Download(NStudyID = kwargs["NStudyID"],
                         NSeriesID = kwargs["NSeriesIDs"][k],
                         dest = DLPath,
                         mode = "all")
            except Exception as e:
                if verbose:
                    print("Download failed. Errmsg: "+str(repr(e)))

        _, file_names, series_ids = GatherSeriesMetadataFromStudy(data_root = str(DLPath),
                                                                  known_metas = kwargs["known_metas"],
                                                                  verbose = verbose)
    else:
        file_names = kwargs["file_names"]
        series_ids = kwargs["series_ids"]

    # If split mode, figure out network/mapfile, then predict, else predict with default network/mapfile
    if split_mode:
        mapfile = kwargs["mapfiles"][str(STModa)]
        network = kwargs["networks"][str(STModa)]

    netw_mask = []
    netw_probs = []
    netw_votes = []

    for m in range(len(series_ids)):
        try:
            # Default series descriptions for emergencies
            if not "Series Description" in list(meta_dict.keys()):
                meta_dict["Series Description"] = []
                for i in range(len(meta_dict["Series Modality"])):
                    meta_dict["Series Description"].append(meta("Series Description", "None"))
            eligibility, probability, prediction = PredictSeriesWithNetwork(STModa = STModa,
                                                                            SEDesc = meta_dict["Series Description"][m].value,
                                                                            SEModa = SEModas[m].value,
                                                                            SEFN = file_names[m],
                                                                            SEID = series_ids[m],
                                                                            mapfile = mapfile,
                                                                            network = network,
                                                                            verbose = verbose,
                                                                            **kwargs)
            netw_mask.append(eligibility)
            netw_probs.append(probability)
            netw_votes.append(prediction)
        except:
            if verbose:
                print("Exception in NetworkPrediction for series "+str(series_ids[m]))
                raise
    if verbose:
        print("IsEligible: "+str(netw_mask))
        print("Netw Probs: "+str(netw_probs))
        print("Netw Votes: "+str(netw_votes))

    weights = [prob if prob >= kwargs["netw_conf_threshold"] else 0 for prob in netw_probs]

    # Apply voting rules that are not rtype "!". (This is weaker than the alternative, but with enough
    # series you will eventually mispredict into WB or H or something and then you lose the whole advantage
    # that series voting offered, that's sadly just statistics)
    corr_votes = netw_votes
    if "vote_rules" in kwargs and "network_vote_rules" in kwargs:
        if kwargs["network_vote_rules"] == True:
            if verbose:
                print("Applying voting rules...")
            for rule in kwargs["vote_rules"]:
                if rule.rtype != "!":
                    corr_votes, r_is_absolute, r_target = rule.apply_to(corr_votes)
            if verbose:
                print("=Corrected votes=")
                print("Network Votes:")
                print(corr_votes)

    # Count the votes
    try:
        vote_hist = {}
        for j, vote in enumerate(corr_votes):
            if netw_mask[j] == 1 and vote:
                if vote in list(vote_hist.keys()):
                    vote_hist[vote] += weights[j]
                else:
                    vote_hist[vote] = weights[j]
        winners = []
        most_votes = max(vote_hist.values())
        winners = [key for i, key in enumerate(list(vote_hist.keys())) if list(vote_hist.values())[i] == most_votes]
        if verbose:
            print(vote_hist, most_votes, winners)
    except:
        if verbose:
            print("Cannot count network series votes.")
        prediction = "UNKNOWN-"+str(STModa)
        decider = "Network failed to make a prediction due to an exception."
    if len(winners) == 0:
        if verbose:
            print("No votes were cast. No eligible series ("+str(valid_modas)+") found or they were no help during substring matching.")
        decider = "Network failed to tiebreak (no votes cast or threshold not met), no prediction will be made."
        prediction = "UNKNOWN-"+str(STModa)
    if len(winners) == 1:
        decider = "Neural network"
        prediction = winners[0]
        return prediction, decider
    if len(winners) >1:
        if verbose:
            print("Multiple predictions have same probability. No decision is made.")
        decider = "Network failed to tiebreak (no votes cast or threshold not met), no prediction will be made."
        prediction = "UNKNOWN-"+str(STModa)
    
    ####################################################################################################
    # Layer 5
    ####################################################################################################
    # Series Voting starts here, try matching as above and then vote by simple majority for a prediction
    if verbose:
        print("Beginning series voting...")
    decider = "Series Voting"

    preliminary_votes = []
    mask = []
    for i in range(len(meta_dict["Procedure Code"])):
        eligibility, votes = PredictSeries(meta_dict = meta_dict,
                                           sidx = i,
                                           STModa = STModa,
                                           mapping = mapping,
                                           verbose = verbose,
                                           **kwargs)
        preliminary_votes.extend(votes)
        mask.append(eligibility)
    
    if verbose:
        print("=Preliminary votes=")
        kc = 0
        for key in sorted(list(meta_dict.keys())):
            if not key in ["Procedure Code", "Study Description", "Series Modality"]:
                print(str(key)+" Votes:")
                print(preliminary_votes[kc::(len(meta_dict)-3)])
                kc += 1
        print("Is eligible:")
        print(mask)
        
    # Apply voting rules (if a rule establishes an absolute, exit rules application. (All legal votes have been set
    # to that absolute value anyway, meaning we can just count votes as we normally would and the absolute wins.)
    if "vote_rules" in kwargs:
        if verbose:
            print("Applying voting rules...")
        combined_votes = preliminary_votes
        for rule in kwargs["vote_rules"]:
            combined_votes, r_is_absolute, r_target = rule.apply_to(combined_votes)
            if r_is_absolute:
                break
        if verbose:
            print("=Corrected votes=")
            kc = 0
            for key in sorted(list(meta_dict.keys())):
                if not key in ["Procedure Code", "Study Description", "Series Modality"]:
                    print(str(key)+" Votes:")
                    print(combined_votes[kc::(len(meta_dict)-3)])
                    kc += 1
            print("Is eligible:")
            print(mask)
    
    # Count all votes, if the vote is not None and the voting series was deemed eligible
    winners = []
    try:
        vote_hist = Counter([vote for i,vote in enumerate(combined_votes) if (mask[int(np.floor(i/(len(meta_dict)-3)))] == 1 and vote)])
        most_votes = max(vote_hist.values())
        winners = [key for i, key in enumerate(list(vote_hist.keys())) if list(vote_hist.values())[i] == most_votes]
        if verbose:
            print(vote_hist, most_votes, winners)
    except:
        if verbose:
            print("Cannot count series votes.")
    if len(winners) == 0:
        if verbose:
            print("No votes were cast. No eligible series ("+str(valid_modas)+") found or they were no help during substring matching.")
        prediction = "UNKNOWN-"+str(STModa)
        decider = "Series vote could not make a prediction."
        return prediction, decider
    if len(winners) == 1:
        return winners[0], decider
    if len(winners) > 1:
        prediction = "UNKNOWN-"+str(STModa)
        decider = "Series vote predicted multiple classes with same number of votes."
        return prediction, decider
    
def PredictStudy_9(meta_dict, mapfile, network, verbose=False, local=True, split_mode=False, **kwargs):
    '''
    As PredictStudy_5, except the NN is merged into layer 4 (4+5=9).
    '''
    
    # Unpack critical metadata
    try:
        Code = meta_dict["Procedure Code"][0].value
        STDesc = meta_dict["Study Description"][0].value
        SEModas = [item for item in meta_dict["Series Modality"]]
    except KeyError as k:
        print("meta_dict is missing critical metadata key: "+str(k)+". Study will not be predicted.")
        print("If this error shows up constantly, the key was probably excluded in the config.ini.")
        print("If this is the case, consider restoring it.")
        raise
        
    # Valid modalities
    with open(mapfile) as json_file:
        mapping = js.load(json_file)
    valid_modas = []
    for moda in list(mapping["Internal"]["Moda"].values()):
        if moda not in valid_modas and moda==moda:
            valid_modas.append(moda)
            
    # Remapped modalities (pretend all series of type X are actually type Y, generally only for debugging)
    if "remapped_modalities" in kwargs:
        if verbose:
            print("Remapping modalities ...")
        SEModas = [SEM if SEM.value not in kwargs["remapped_modalities"] else meta("Series Modality", kwargs["remapped_modalities"][str(SEM.value)]) for SEM in SEModas]

    # Rule-based approach to Modality of Study
    if any(SEModas):
        try:
            tmp = [m.value for m in SEModas if m.value in valid_modas]
            hist = (Counter(tmp))
            STModa = max(hist, key=hist.get)
        except:
            STModa = ""
    else:
        STModa = ""
    if any(x for x in SEModas if x.value == "PT"):
        if STModa == "CT":
            STModa = "PT"
        elif STModa == "MR":
            STModa = "MRPET"
            
    # Classify study as unknown if the modality is entirely unmapped
    if not STModa:
        try:
            tmp = [m.value for m in SEModas if m.value]
            hist = (Counter(tmp))
            STModa = max(hist, key=hist.get)
        except:
            pass
        if not (STModa in valid_modas):
            prediction = "UNKNOWN-"+str(STModa)
            decider = "Unmapped modality provided"
            return prediction, decider
        else:
            prediction = "UNKNOWN"
            decider = "No modality provided"
            return prediction, decider
    
    ####################################################################################################
    # Layer 1
    ####################################################################################################
    if verbose:
        print("Trying to match procedure code.")
    decider = "Code, Full Match"
    try:
        c = list(mapping["Internal"]["Code"].keys())[list(mapping["Internal"]["Code"].values()).index(str(Code))]
        prediction = mapping["Internal"]["Code"][str(c)]
        if verbose:
            print("Full match for procedure code found.")
        return prediction, decider
    except:
        # No match for "Code", try Multiclass
        try:
            prediction = mapping["Internal"]["Multiclass"][str(Code)]
            if verbose:
                print("Full match for procedure code found.")
            return prediction, decider
        except:
            # No matches
            if verbose:
                print("No exact match for Procedure Code in mapping.")
    ####################################################################################################
    # Layer 2
    ####################################################################################################
    if verbose:
        print("Trying to match study description.")
    decider = "Reduced Study Description, Full Match"
    try:
        c = Reduce(list(mapping["Internal"]["Desc"].keys()))[Reduce(list(mapping["Internal"]["Desc"].values())).index(str(Reduce(STDesc)))]
        prediction = mapping["Internal"]["Code"][str(c)]
        # Only accept full match if modality is correct (unless no modality is given, in which case, accept all)
        if mapping["Internal"]["Moda"][str(c)] == STModa or not STModa:
            if verbose:
                print("Full match for Study Description found.")
            return prediction, decider
        else:
            if verbose:
                print("Exact match for study description in mapping, but wrong modality.")
    except:
        # No exact match
        if verbose:
            print("No exact match for study description in mapping.")
    ####################################################################################################
    # Layer 3
    ####################################################################################################
    if verbose:
        print("Trying to match substrings from collected metadata.")
    decider = "Voting (1 vote each best substring match over all collected metadata)"
    
    # Grab all descriptions from mapping which are legal, based on known modality
    # If modality is unknown or not one of the main ones, allow all modalities
    # Grab all descriptions from mapping which are legal, based on known modality
    if STModa == "MRPET":
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
    elif STModa == "MR":
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
    elif not STModa:
        # Making sure to exclude the nan variant from the JSON
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if key==key]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if val==val]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if key==key]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if val==val]
    else:
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
    
    # If there is only one item in keys, we do not need to match anything, this is the only possibility
    # Typically, this should only ever apply to MG, but who knows. If this is incorrect, then the mapping
    # needs to be adjusted.
    desclen = len(keys)/2
    if desclen == 1:
        stdesc_prediction = mapping["Internal"]["Code"][str(keys[0])]
        if verbose:
            print("Modality has only one entry in mapping, this should be the correct class.")
        return stdesc_prediction, decider
    
    # Add the modality to all keywords, and once at the end for description, add PT as PETCT
    if STModa == "PT":
        ML = ["PT", "PET CT", "CT PET"]
    elif STModa == "MR":
        ML = ["MRT", "MR"]
    elif STModa == "MRPET":
        ML = ["MR PET", "PET MR", "MRT PET", "PET MRT"]
    else:
        ML = [STModa]
    for idx, alts in enumerate(vals):
        if idx < desclen:
            vals[idx] = str(";")+";".join([vals[idx]+str(";")+vals[idx]+M for M in ML])+str(";")
        else:
            vals[idx] = str(";")+";".join(";".join([item+M+str(";")+M+item+str(";")+item for item in alts.split(";")]) for M in ML)+str(";")

    # Substring matching
    if "minmatchlength" in kwargs:
        meml = kwargs["minmatchlength"]["study"][0]
        mrml = kwargs["minmatchlength"]["study"][0]
    else:
        meml = 4
        mrml = 6
    stdesc_prediction, is_exact = SubstringMatcher(keys = keys,
                                                   vals = Reduce(vals),
                                                   desc = STDesc, 
                                                   meml = meml, 
                                                   mrml = mrml, 
                                                   mapping = mapping, 
                                                   verbose = verbose, 
                                                   return_exact = True, 
                                                   **kwargs)
            
    # Instant hits - If we score an exact match in the study description, there is no reason to continue and
    # we immediately return the value
    if is_exact:
        decider = "Best substring match for Study Description"
        if verbose:
            print("Study description had exact match, this is our best guess.")
        return stdesc_prediction, decider
    
    ####################################################################################################
    # Layer 4/9
    ####################################################################################################
    # Series Voting starts here, try matching as above and then vote by simple majority for a prediction
    # Additionally, we merge the network votes into the series voting, combining their knowledge
    if verbose:
        print("Beginning series voting...")
    decider = "Merged Voting (Metadata + Network)"

    preliminary_votes = []
    mask = []
    for i in range(len(meta_dict["Procedure Code"])):
        eligibility, votes = PredictSeries(meta_dict = meta_dict,
                                           sidx = i,
                                           STModa = STModa,
                                           mapping = mapping,
                                           verbose = verbose,
                                           **kwargs)
        preliminary_votes.extend(votes)
        mask.append(eligibility)
    
    if verbose:
        print("=Preliminary votes=")
        kc = 0
        for key in sorted(list(meta_dict.keys())):
            if not key in ["Procedure Code", "Study Description", "Series Modality"]:
                print(str(key)+" Votes:")
                print(preliminary_votes[kc::(len(meta_dict)-3)])
                kc += 1
        print("Is eligible:")
        print(mask)
        
    # Apply voting rules (if a rule establishes an absolute, exit rules application. (All legal votes have been set
    # to that absolute value anyway, meaning we can just count votes as we normally would and the absolute wins.)
    if "vote_rules" in kwargs:
        if verbose:
            print("Applying voting rules...")
        combined_votes = preliminary_votes
        for rule in kwargs["vote_rules"]:
            combined_votes, r_is_absolute, r_target = rule.apply_to(combined_votes)
            if r_is_absolute:
                break
        if verbose:
            print("=Corrected votes=")
            kc = 0
            for key in sorted(list(meta_dict.keys())):
                if not key in ["Procedure Code", "Study Description", "Series Modality"]:
                    print(str(key)+" Votes:")
                    print(combined_votes[kc::(len(meta_dict)-3)])
                    kc += 1
            print("Is eligible:")
            print(mask)
    
    # If we caught an absolute rule, no need to count further votes. This should by all accounts be right.
    if r_is_absolute:
        if verbose:
            print("Caught rtype '!', exiting.")
        return r_target, decider
    
    # Count all votes, if the vote is not None and the voting series was deemed eligible
    winners = []
    try:
        vote_hist = Counter([vote for i,vote in enumerate(combined_votes) if (mask[int(np.floor(i/(len(meta_dict)-3)))] == 1 and vote)])
    except:
        if verbose:
            print("Cannot count series votes, exiting with no prediction.")
        prediction = "UNKNOWN-"+str(STModa)
        return prediction, decider
    
    if "no_network" not in kwargs or ("no_network" in kwargs and kwargs["no_network"] == False):
        # If predictions are made locally, the files need not be downloaded - non-local files must be acquired
        # first, else the network has nothing to evaluate on
        if not local:
            if verbose:
                print("Acquiring data ...")
            # Remove potential leftovers
            DLPath = './tempeval/'
            shutil.rmtree(DLPath)
            os.makedirs(DLPath, exist_ok=True)
            # Download files
            for k in range(len(kwargs["NSeriesIDs"])):
                try:
                    Download(NStudyID = kwargs["NStudyID"],
                             NSeriesID = kwargs["NSeriesIDs"][k],
                             dest = DLPath,
                             mode = "all")
                except Exception as e:
                    if verbose:
                        print("Download failed. Errmsg: "+str(repr(e)))

            _, file_names, series_ids = GatherSeriesMetadataFromStudy(data_root = str(DLPath),
                                                                      known_metas = kwargs["known_metas"],
                                                                      verbose = verbose)
        else:
            file_names = kwargs["file_names"]
            series_ids = kwargs["series_ids"]

        # If split mode, figure out network/mapfile, then predict, else predict with default network/mapfile
        if split_mode:
            mapfile = kwargs["mapfiles"][str(STModa)]
            network = kwargs["networks"][str(STModa)]

        netw_mask = []
        netw_probs = []
        netw_votes = []

        for m in range(len(series_ids)):
            try:
                # Default series descriptions for emergencies
                if not "Series Description" in list(meta_dict.keys()):
                    meta_dict["Series Description"] = []
                    for i in range(len(meta_dict["Series Modality"])):
                        meta_dict["Series Description"].append(meta("Series Description", "None"))
                eligibility, probability, prediction = PredictSeriesWithNetwork(STModa = STModa,
                                                                                SEDesc = meta_dict["Series Description"][m].value,
                                                                                SEModa = SEModas[m].value,
                                                                                SEFN = file_names[m],
                                                                                SEID = series_ids[m],
                                                                                mapfile = mapfile,
                                                                                network = network,
                                                                                verbose = verbose,
                                                                                **kwargs)
                netw_mask.append(eligibility)
                netw_probs.append(probability)
                netw_votes.append(prediction)
            except:
                if verbose:
                    print("Exception in NetworkPrediction for series "+str(series_ids[m]))
                    raise
        if verbose:
            print("IsEligible: "+str(netw_mask))
            print("Netw Probs: "+str(netw_probs))
            print("Netw Votes: "+str(netw_votes))

        # Add network predictions on top of original votes, weighted by the probability assigned by the
        # network, apply voting rules again, then count votes and potentially make a decision
        weights = [prob if prob >= kwargs["netw_conf_threshold"] else 0 for prob in netw_probs]
        nvn = len(netw_votes)

        # Reapply voting rules, excluding rtype "!". (This is weaker than the alternative, but with enough
        # series you will eventually mispredict into WB or H or something and then you lose the whole advantage
        # that series voting offered)
        # If we are allowed to use vote_rules on network votes, we combine with the preliminary votes from before
        # and apply the rules to the combination. If not, we stack the network votes on top of the modified
        # votes from before and only do the recount.
        if "vote_rules" in kwargs and "network_vote_rules" in kwargs:
            if kwargs["network_vote_rules"] == True:
                combined_votes = netw_votes + preliminary_votes
                nvn = len(netw_votes)
                if verbose:
                    print("Applying voting rules...")
                for rule in kwargs["vote_rules"]:
                    if rule.rtype != "!":
                        combined_votes, r_is_absolute, r_target = rule.apply_to(combined_votes)
                if verbose:
                    print("=Corrected votes=")
                    kc = 0
                    for key in sorted(list(meta_dict.keys())):
                        if not key in ["Procedure Code", "Study Description", "Series Modality"]:
                            print(str(key)+" Votes:")
                            print(combined_votes[nvn+kc::(len(meta_dict)-3)])
                            kc += 1
                    print("Network Votes:")
                    print(combined_votes[0:nvn])
            else:
                combined_votes = netw_votes + combined_votes
        else:
            combined_votes = netw_votes + combined_votes

        # Count all votes again (and count kc if that was not done before)
        kc = 0
        for key in sorted(list(meta_dict.keys())):
            if not key in ["Procedure Code", "Study Description", "Series Modality"]:
                kc += 1
        try:
            for j, vote in enumerate(combined_votes[0:nvn]):
                if netw_mask[j] == 1 and vote:
                    if vote in list(vote_hist.keys()):
                        vote_hist[vote] += weights[j]
                    else:
                        vote_hist[vote] = weights[j]
            winners = []
            most_votes = max(vote_hist.values())
            winners = [key for i, key in enumerate(list(vote_hist.keys())) if list(vote_hist.values())[i] == most_votes]
            if verbose:
                print(vote_hist, most_votes, winners)
        except:
            if verbose:
                print("Cannot count network series votes.")
            prediction = "UNKNOWN-"+str(STModa)
            decider = "Network failed to make a prediction due to an exception."
            raise #Disable for prod
            return prediction, decider
        if len(winners) == 0:
            if verbose:
                print("No votes were cast. No eligible series ("+str(valid_modas)+") found or they were no help during substring matching.")
            decider = "Network failed to tiebreak (no votes cast or threshold not met), no prediction will be made."
            prediction = "UNKNOWN-"+str(STModa)
            return prediction, decider
        if len(winners) == 1:
            prediction = winners[0]
            return prediction, decider
        if len(winners) > 1:
            if verbose:
                print("Multiple predictions have same probability.")
            decider = "Network failed to tiebreak (no votes cast or threshold not met), no prediction will be made."
            prediction = "UNKNOWN-"+str(STModa)
            return prediction, decider
        
def PredictStudy_12(meta_dict, mapfile, network, verbose=False, local=True, split_mode=False, **kwargs):
    '''
    As PredictStudy_5, except the NN is merged into layers 3+4 (3+4+5=12).
    '''
    
    # Unpack critical metadata
    try:
        Code = meta_dict["Procedure Code"][0].value
        STDesc = meta_dict["Study Description"][0].value
        SEModas = [item for item in meta_dict["Series Modality"]]
    except KeyError as k:
        print("meta_dict is missing critical metadata key: "+str(k)+". Study will not be predicted.")
        print("If this error shows up constantly, the key was probably excluded in the config.ini.")
        print("If this is the case, consider restoring it.")
        raise
        
    # Valid modalities
    with open(mapfile) as json_file:
        mapping = js.load(json_file)
    valid_modas = []
    for moda in list(mapping["Internal"]["Moda"].values()):
        if moda not in valid_modas and moda==moda:
            valid_modas.append(moda)
            
    # Remapped modalities (pretend all series of type X are actually type Y, generally only for debugging)
    if "remapped_modalities" in kwargs:
        if verbose:
            print("Remapping modalities ...")
        SEModas = [SEM if SEM.value not in kwargs["remapped_modalities"] else meta("Series Modality", kwargs["remapped_modalities"][str(SEM.value)]) for SEM in SEModas]

    # Rule-based approach to Modality of Study
    if any(SEModas):
        try:
            tmp = [m.value for m in SEModas if m.value in valid_modas]
            hist = (Counter(tmp))
            STModa = max(hist, key=hist.get)
        except:
            STModa = ""
    else:
        STModa = ""
    if any(x for x in SEModas if x.value == "PT"):
        if STModa == "CT":
            STModa = "PT"
        elif STModa == "MR":
            STModa = "MRPET"
            
    # Classify study as unknown if the modality is entirely unmapped
    if not STModa:
        try:
            tmp = [m.value for m in SEModas if m.value]
            hist = (Counter(tmp))
            STModa = max(hist, key=hist.get)
        except:
            pass
        if not (STModa in valid_modas):
            prediction = "UNKNOWN-"+str(STModa)
            decider = "Unmapped modality provided"
            return prediction, decider
        else:
            prediction = "UNKNOWN"
            decider = "No modality provided"
            return prediction, decider
    
    ####################################################################################################
    # Layer 1
    ####################################################################################################
    if verbose:
        print("Trying to match procedure code.")
    decider = "Code, Full Match"
    try:
        c = list(mapping["Internal"]["Code"].keys())[list(mapping["Internal"]["Code"].values()).index(str(Code))]
        prediction = mapping["Internal"]["Code"][str(c)]
        if verbose:
            print("Full match for procedure code found.")
        return prediction, decider
    except:
        # No match for "Code", try Multiclass
        try:
            prediction = mapping["Internal"]["Multiclass"][str(Code)]
            if verbose:
                print("Full match for procedure code found.")
            return prediction, decider
        except:
            # No matches
            if verbose:
                print("No exact match for Procedure Code in mapping.")
    ####################################################################################################
    # Layer 2
    ####################################################################################################
    if verbose:
        print("Trying to match study description.")
    decider = "Reduced Study Description, Full Match"
    try:
        c = Reduce(list(mapping["Internal"]["Desc"].keys()))[Reduce(list(mapping["Internal"]["Desc"].values())).index(str(Reduce(STDesc)))]
        prediction = mapping["Internal"]["Code"][str(c)]
        # Only accept full match if modality is correct (unless no modality is given, in which case, accept all)
        if mapping["Internal"]["Moda"][str(c)] == STModa or not STModa:
            if verbose:
                print("Full match for Study Description found.")
            return prediction, decider
        else:
            if verbose:
                print("Exact match for study description in mapping, but wrong modality.")
    except:
        # No exact match
        if verbose:
            print("No exact match for study description in mapping.")
    ####################################################################################################
    # Layer 3/12
    ####################################################################################################
    if verbose:
        print("Trying to match substrings from collected metadata.")
    decider = "Voting (1 vote each best substring match over all collected metadata)"
    
    # Grab all descriptions from mapping which are legal, based on known modality
    # If modality is unknown or not one of the main ones, allow all modalities
    # Grab all descriptions from mapping which are legal, based on known modality
    if STModa == "MRPET":
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if "MR PET" in str(mapping["Internal"]["Desc"][str(i)])]
    elif STModa == "MR":
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if "MR PET" not in str(mapping["Internal"]["Desc"][str(i)]) and STModa == str(mapping["Internal"]["Moda"][str(i)])]
    elif not STModa:
        # Making sure to exclude the nan variant from the JSON
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if key==key]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if val==val]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if key==key]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if val==val]
    else:
        keys = [key for i, key in enumerate(mapping["Internal"]["Desc"].keys()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals = [val for i, val in enumerate(mapping["Internal"]["Desc"].values()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
        keys += [key for i, key in enumerate(mapping["Internal"]["Alts"].keys()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
        vals += [val for i, val in enumerate(mapping["Internal"]["Alts"].values()) if STModa == str(mapping["Internal"]["Moda"][str(i)])]
    
    # If there is only one item in keys, we do not need to match anything, this is the only possibility
    # Typically, this should only ever apply to MG, but who knows. If this is incorrect, then the mapping
    # needs to be adjusted.
    desclen = len(keys)/2
    if desclen == 1:
        stdesc_prediction = mapping["Internal"]["Code"][str(keys[0])]
        if verbose:
            print("Modality has only one entry in mapping, this should be the correct class.")
        return stdesc_prediction, decider
    
    # Add the modality to all keywords, and once at the end for description, add PT as PETCT
    if STModa == "PT":
        ML = ["PT", "PET CT", "CT PET"]
    elif STModa == "MR":
        ML = ["MRT", "MR"]
    elif STModa == "MRPET":
        ML = ["MR PET", "PET MR", "MRT PET", "PET MRT"]
    else:
        ML = [STModa]
    for idx, alts in enumerate(vals):
        if idx < desclen:
            vals[idx] = str(";")+";".join([vals[idx]+str(";")+vals[idx]+M for M in ML])+str(";")
        else:
            vals[idx] = str(";")+";".join(";".join([item+M+str(";")+M+item+str(";")+item for item in alts.split(";")]) for M in ML)+str(";")

    # Substring matching
    if "minmatchlength" in kwargs:
        meml = kwargs["minmatchlength"]["study"][0]
        mrml = kwargs["minmatchlength"]["study"][0]
    else:
        meml = 4
        mrml = 6
    stdesc_prediction, is_exact = SubstringMatcher(keys = keys,
                                                   vals = Reduce(vals),
                                                   desc = STDesc, 
                                                   meml = meml, 
                                                   mrml = mrml, 
                                                   mapping = mapping, 
                                                   verbose = verbose, 
                                                   return_exact = True, 
                                                   **kwargs)
            
    # Instant hits - If we score an exact match in the study description, we weight it strongly. If not,
    # we give it a lower impact. Also test for priority matches.
    if is_exact:
        stdesc_weight = kwargs["a12_gmw"]*len(meta_dict["Procedure Code"])
    else:
        stdesc_weight = kwargs["a12_wmw"]*len(meta_dict["Procedure Code"])
    for rule in kwargs["vote_rules"]:
        _, r_is_absolute, r_target = rule.apply_to([stdesc_prediction])
        if r_is_absolute:
                break
    if r_is_absolute:
        decider = "Study description matches rule with rtype '!', exiting."
        if verbose:
            print("Caught rtype '!', exiting.")
        return r_target, decider
    
    # Series Voting starts here, try matching as above and then vote by simple majority for a prediction
    # Additionally, we merge the network votes into the series voting, combining their knowledge
    if verbose:
        print("Beginning series voting...")
    decider = "Merged Voting (Metadata + Network)"

    preliminary_votes = []
    mask = []
    for i in range(len(meta_dict["Procedure Code"])):
        eligibility, votes = PredictSeries(meta_dict = meta_dict,
                                           sidx = i,
                                           STModa = STModa,
                                           mapping = mapping,
                                           verbose = verbose,
                                           **kwargs)
        preliminary_votes.extend(votes)
        mask.append(eligibility)
    
    if verbose:
        print("=Preliminary votes=")
        kc = 0
        for key in sorted(list(meta_dict.keys())):
            if not key in ["Procedure Code", "Study Description", "Series Modality"]:
                print(str(key)+" Votes:")
                print(preliminary_votes[kc::(len(meta_dict)-3)])
                kc += 1
        print("Is eligible:")
        print(mask)
        
    # Apply voting rules (if a rule establishes an absolute, exit rules application. (All legal votes have been set
    # to that absolute value anyway, meaning we can just count votes as we normally would and the absolute wins.)
    if "vote_rules" in kwargs:
        if verbose:
            print("Applying voting rules...")
        combined_votes = preliminary_votes
        for rule in kwargs["vote_rules"]:
            combined_votes, r_is_absolute, r_target = rule.apply_to(combined_votes)
            if r_is_absolute:
                break
        if verbose:
            print("=Corrected votes=")
            kc = 0
            for key in sorted(list(meta_dict.keys())):
                if not key in ["Procedure Code", "Study Description", "Series Modality"]:
                    print(str(key)+" Votes:")
                    print(combined_votes[kc::(len(meta_dict)-3)])
                    kc += 1
            print("Is eligible:")
            print(mask)
    
    # If we caught an absolute rule, no need to count further votes. This should by all accounts be right.
    if r_is_absolute:
        if verbose:
            print("Caught rtype '!', exiting.")
        return r_target, decider
    
    # Count all votes, if the vote is not None and the voting series was deemed eligible
    winners = []
    try:
        vote_hist = Counter([vote for i,vote in enumerate(combined_votes) if (mask[int(np.floor(i/(len(meta_dict)-3)))] == 1 and vote)])
    except:
        if verbose:
            print("Cannot count series votes, exiting with no prediction.")
        prediction = "UNKNOWN-"+str(STModa)
        return prediction, decider
    
    if "no_network" not in kwargs or ("no_network" in kwargs and kwargs["no_network"] == False):
        # If predictions are made locally, the files need not be downloaded - non-local files must be acquired
        # first, else the network has nothing to evaluate on
        if not local:
            if verbose:
                print("Acquiring data ...")
            # Remove potential leftovers
            DLPath = './tempeval/'
            shutil.rmtree(DLPath)
            os.makedirs(DLPath, exist_ok=True)
            # Download files
            for k in range(len(kwargs["NSeriesIDs"])):
                try:
                    Download(NStudyID = kwargs["NStudyID"],
                             NSeriesID = kwargs["NSeriesIDs"][k],
                             dest = DLPath,
                             mode = "all")
                except Exception as e:
                    if verbose:
                        print("Download failed. Errmsg: "+str(repr(e)))

            _, file_names, series_ids = GatherSeriesMetadataFromStudy(data_root = str(DLPath),
                                                                      known_metas = kwargs["known_metas"],
                                                                      verbose = verbose)
        else:
            file_names = kwargs["file_names"]
            series_ids = kwargs["series_ids"]

        # If split mode, figure out network/mapfile, then predict, else predict with default network/mapfile
        if split_mode:
            mapfile = kwargs["mapfiles"][str(STModa)]
            network = kwargs["networks"][str(STModa)]

        netw_mask = []
        netw_probs = []
        netw_votes = []

        for m in range(len(series_ids)):
            try:
                # Default series descriptions for emergencies
                if not "Series Description" in list(meta_dict.keys()):
                    meta_dict["Series Description"] = []
                    for i in range(len(meta_dict["Series Modality"])):
                        meta_dict["Series Description"].append(meta("Series Description", "None"))
                eligibility, probability, prediction = PredictSeriesWithNetwork(STModa = STModa,
                                                                                SEDesc = meta_dict["Series Description"][m].value,
                                                                                SEModa = SEModas[m].value,
                                                                                SEFN = file_names[m],
                                                                                SEID = series_ids[m],
                                                                                mapfile = mapfile,
                                                                                network = network,
                                                                                verbose = verbose,
                                                                                **kwargs)
                netw_mask.append(eligibility)
                netw_probs.append(probability)
                netw_votes.append(prediction)
            except:
                if verbose:
                    print("Exception in NetworkPrediction for series "+str(series_ids[m]))
                    raise
        if verbose:
            print("IsEligible: "+str(netw_mask))
            print("Netw Probs: "+str(netw_probs))
            print("Netw Votes: "+str(netw_votes))

        # Add network predictions on top of original votes, weighted by the probability assigned by the
        # network, apply voting rules again, then count votes and potentially make a decision
        weights = [prob if prob >= kwargs["netw_conf_threshold"] else 0 for prob in netw_probs]
        nvn = len(netw_votes)

        # Reapply voting rules, excluding rtype "!". (This is weaker than the alternative, but with enough
        # series you will eventually mispredict into WB or H or something and then you lose the whole advantage
        # that series voting offered)
        # If we are allowed to use vote_rules on network votes, we combine with the preliminary votes from before
        # and apply the rules to the combination. If not, we stack the network votes on top of the modified
        # votes from before and only do the recount.
        if "vote_rules" in kwargs and "network_vote_rules" in kwargs:
            if kwargs["network_vote_rules"] == True:
                combined_votes = netw_votes + preliminary_votes
                nvn = len(netw_votes)
                if verbose:
                    print("Applying voting rules...")
                for rule in kwargs["vote_rules"]:
                    if rule.rtype != "!":
                        combined_votes, r_is_absolute, r_target = rule.apply_to(combined_votes)
                if verbose:
                    print("=Corrected votes=")
                    kc = 0
                    for key in sorted(list(meta_dict.keys())):
                        if not key in ["Procedure Code", "Study Description", "Series Modality"]:
                            print(str(key)+" Votes:")
                            print(combined_votes[nvn+kc::(len(meta_dict)-3)])
                            kc += 1
                    print("Network Votes:")
                    print(combined_votes[0:nvn])
            else:
                combined_votes = netw_votes + combined_votes
        else:
            combined_votes = netw_votes + combined_votes

        # Count all votes again (and count kc if that was not done before)
        kc = 0
        for key in sorted(list(meta_dict.keys())):
            if not key in ["Procedure Code", "Study Description", "Series Modality"]:
                kc += 1
        try:
            for j, vote in enumerate(combined_votes[0:nvn]):
                if netw_mask[j] == 1 and vote:
                    if vote in list(vote_hist.keys()):
                        vote_hist[vote] += weights[j]
                    else:
                        vote_hist[vote] = weights[j]
            # Add the study description "votes"
            if stdesc_prediction:
                if stdesc_prediction in list(vote_hist.keys()):
                    vote_hist[stdesc_prediction] += stdesc_weight
                else:
                    vote_hist[stdesc_prediction] = stdesc_weight
            # Finally count    
            winners = []
            most_votes = max(vote_hist.values())
            winners = [key for i, key in enumerate(list(vote_hist.keys())) if list(vote_hist.values())[i] == most_votes]
            if verbose:
                print(vote_hist, most_votes, winners)
        except:
            if verbose:
                print("Cannot count network series votes.")
            prediction = "UNKNOWN-"+str(STModa)
            decider = "Network failed to make a prediction due to an exception."
            raise #Disable for prod
            return prediction, decider
        if len(winners) == 0:
            if verbose:
                print("No votes were cast. No eligible series ("+str(valid_modas)+") found or they were no help during substring matching.")
            decider = "Network failed to tiebreak (no votes cast or threshold not met), no prediction will be made."
            prediction = "UNKNOWN-"+str(STModa)
            return prediction, decider
        if len(winners) == 1:
            prediction = winners[0]
            return prediction, decider
        if len(winners) > 1:
            if verbose:
                print("Multiple predictions have same probability.")
            decider = "Network failed to tiebreak (no votes cast or threshold not met), no prediction will be made."
            prediction = "UNKNOWN-"+str(STModa)
            return prediction, decider

# End-to-end predictions
        
def CDtoPrediction(data_root, known_metas, mapfile, network, verbose=False, recoverfrompi=False, algo=5, **kwargs):
    # Returns a class prediction for a study, given a root directory, based on a mapfile and a network,
    # and an arbitrary number of entries in the DICOM header which will be tried for matches.
    # Additionally returns the decider for the prediction (or the netw_votes dictionary if only the network
    # is used for predictions, since here its clear anyway that the network must be the decider).
    # Kwargs: All kwargs in PredictStudy (as every kwarg is handed down the chain)
    #         no_network (if set, disables network tiebreaks entirely for testing purposes)
    # Exit codes: 0 (success), 1 (error), 2 (no prediction)
    try:
        mdict, SEFNs, SEIDs = GatherSeriesMetadataFromStudy(data_root = data_root,
                                                            known_metas = known_metas, 
                                                            verbose = verbose)
    except Exception as e:
        if verbose:
            print("Could not gather any metadata, probably file(s) not found.")
        exitcode = 1
        prediction = "UNKNOWN"
        decider = "Error while gathering metadata"
        return exitcode, (prediction, decider), e
    try:
        # If required, replace the study description entries entirely, replacing them all with the one recovered
        # from the database where the original one is kept.
        if recoverfrompi:
            if verbose:
                print("Recovered original study description from PACS: "+str(kwargs["RecoveredStudyDescription"]))
            mdict["Study Description"] = []
            while len(mdict["Study Description"]) < len(mdict["Procedure Code"]):
                mdict["Study Description"].append(meta("Study Decription", kwargs["RecoveredStudyDescription"]))
        
        if algo not in [0,1,2,3,4,5,9,12]:
            raise ValueError("Choose an existing algorithm (0, 2, 3, 4, 5, 9, 12), please.")
        if algo==1:
            raise ValueError("Algorithm 1 is largely pointless, given that the network can predict everything the metadata-algorithm can, if it gets the chance to. Therefore, Algo 1 and 0 are quasi-identical, barring further changes to the program. Hence, I opted not to implement it.")
        if algo in [0,2,3,4] and verbose:
            print("The Keywordarg no_network is disabled for the ablation study (the reasons should be obvious. I can't test network influence if it is disabled.)")
        
        # Decide on Predictor, Algo==N means CNN as layer N, with the special case 0 being one where the entire
        # prediction is ONLY network (plus voting rules) based, or 9, where layers 4 and 5 (one being the network)
        # are merged
        if algo == 12:
            prediction, decider = PredictStudy_12(meta_dict = mdict,
                                                 file_names = SEFNs,
                                                 series_ids = SEIDs,
                                                 mapfile = mapfile,
                                                 network = network,
                                                 local = True,
                                                 verbose = verbose,
                                                 DLPath = data_root,
                                                 **kwargs)
        if algo == 9:
            prediction, decider = PredictStudy_9(meta_dict = mdict,
                                                 file_names = SEFNs,
                                                 series_ids = SEIDs,
                                                 mapfile = mapfile,
                                                 network = network,
                                                 local = True,
                                                 verbose = verbose,
                                                 DLPath = data_root,
                                                 **kwargs)
        
        if algo == 5:
            prediction, decider = PredictStudy_5(meta_dict = mdict,
                                                 file_names = SEFNs,
                                                 series_ids = SEIDs,
                                                 mapfile = mapfile,
                                                 network = network,
                                                 local = True,
                                                 verbose = verbose,
                                                 DLPath = data_root,
                                                 **kwargs)
        if algo == 4:
            prediction, decider = PredictStudy_4(meta_dict = mdict,
                                                 file_names = SEFNs,
                                                 series_ids = SEIDs,
                                                 mapfile = mapfile,
                                                 network = network,
                                                 local = True,
                                                 verbose = verbose,
                                                 DLPath = data_root,
                                                 **kwargs)
        if algo == 3:
            prediction, decider = PredictStudy_3(meta_dict = mdict,
                                                 file_names = SEFNs,
                                                 series_ids = SEIDs,
                                                 mapfile = mapfile,
                                                 network = network,
                                                 local = True,
                                                 verbose = verbose,
                                                 DLPath = data_root,
                                                 **kwargs)
        if algo == 2:
            prediction, decider = PredictStudy_2(meta_dict = mdict,
                                                 file_names = SEFNs,
                                                 series_ids = SEIDs,
                                                 mapfile = mapfile,
                                                 network = network,
                                                 local = True,
                                                 verbose = verbose,
                                                 DLPath = data_root,
                                                 **kwargs)
        if algo == 0:
            prediction, decider = PredictStudy_0(meta_dict = mdict,
                                                 file_names = SEFNs,
                                                 series_ids = SEIDs,
                                                 mapfile = mapfile,
                                                 network = network,
                                                 local = True,
                                                 verbose = verbose,
                                                 DLPath = data_root,
                                                 **kwargs)
        if prediction == None or "UNKNOWN" in prediction:
            exitcode = 2
        else:
            exitcode = 0
        return exitcode, (prediction, decider), None
    except Exception as e:
        if verbose:
            print("Error in predictor!")
            print(repr(e))
        exitcode = 1
        prediction = "UNKNOWN"
        decider = "Error in predictor"
        return exitcode, (prediction, decider), e
