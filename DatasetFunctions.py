#author: Anonymous

import os
import sys
import time
import shutil
import concurrent.futures
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import torchvision.transforms.functional as ttf

import matplotlib.pyplot as plt
import numpy as np
import random
from collections import Counter

import pydicom
import nibabel
import json as js
import pandas as pd
import SimpleITK as sitk

def refuse_garbage_input():
    raise InputError("Input did not compute and was refused. This is either a bug, or because the input was garbage. If you see this message, something has gone terribly wrong!")

def sitk_read(dicom_directory):
    """
    Returns a list of sitk_images, given a folder in which there is at least one DICOM series
    """
    try:
        # If it is an image series, files should be readable as a series
        sitkreader = sitk.ImageSeriesReader()

        # check if there is a DICOM series in the dicom_directory
        series_IDs = sitkreader.GetGDCMSeriesIDs(dicom_directory)
        print ("Loading dicom folder %" + dicom_directory)
        
        sitk_images = []
        print ("Detected "+str(len(series_IDs))+" distinct series. Loading files ...")
        for idx, ID in enumerate(series_IDs):
            # get all file names
            series_file_names = sitkreader.GetGDCMSeriesFileNames(dicom_directory, series_IDs[idx])
            print(str(len(series_file_names))+" files in series. Attempting cleanup if necessary ...")
            file_sizes = []
            
            # try cleaning out garbage from series
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
            print("Cleanup complete. "+str(len(series_file_names))+" files remain in series.") 
            
            # load series
            sitkreader.SetFileNames(series_file_names)
            sitk_image = sitkreader.Execute()
            sitk_images.append(sitk_image)
        print("Loaded.")

        return sitk_images
    
    except Exception as e:
        if not series_IDs:
            print("Given directory \""+dicom_directory+"\" does not contain a DICOM series!")
            raise
        else:
            print("Unknown error in series reader.") 
            raise

def getDirectoryList(path):
    """
    Returns a list of all paths/to/folders, recursively checking the input path and all sub-directories for
    existence of a specific file (here .dcm)
    """
    directoryList = []

    #return nothing if path is a file
    if os.path.isfile(path):
        return []

    #add path to directoryList if it contains .dcm files
    if len([f for f in os.listdir(path) if f.endswith('.dcm')])>0:
        directoryList.append(path)

    for d in os.listdir(path):
        new_path = os.path.join(path, d)
        if os.path.isdir(new_path):
            directoryList += getDirectoryList(new_path)

    return directoryList

def get_class(mapping, c='', d='', nopet=False, ispi=False, m=''):
    '''
    Given a mapping json (or other dictionary file), this function searches for the corresponding class of the
    given combination of code, description and modality. If the correct class can not be determined, it returns
    the highest class available in the mapping JSON (which should be NaN).
    '''
    
    if ispi: # Remove "PI-" from class, remove useless description
        c=c[3:]
        d=''
        
    d = d.replace("ä","ae")
    d = d.replace("ö","oe")
    d = d.replace("ü","ue")
    d = d.replace("Ä","AE")
    d = d.replace("Ö","OE")
    d = d.replace("Ü","UE")
    d = d.replace("ß","ss")
    d = d.replace("/"," ")
    
    # Search mapping for given value, return the corresponding key (the key is the class)
    try:
        c_class = list(mapping["Internal"]["Code"].keys())[list(mapping["Internal"]["Code"].values()).index(str(c))]
    except:
        try:
            cmc = mapping["Internal"]["Multiclass"][str(c)]
            c_class = list(mapping["Internal"]["Code"].keys())[list(mapping["Internal"]["Code"].values()).index(str(cmc))]
        except:
            c_class = ''
    try:
        d_class = list(mapping["Internal"]["Code"].keys())[list(mapping["Internal"]["Desc"].values()).index(str(d))]
    except:
        d_class = ''
        
    # Map PET scans to NaN if nopet is True (which means they will be removed later)
    if nopet and m=="PT":
        return int(len(list(mapping["Internal"]["Code"].keys()))-1)
    
    # Test whether an agreement was found, given known class names and descriptions, and if yes, return int(class)
    if c_class and c_class==d_class:
        #print('agreed', c_class)
        return int(c_class)
    elif c_class:
        #print('c choice', c_class)
        return int(c_class)
    elif d_class:
        #print('d choice', d_class)
        return int(d_class)
    # The last class is mapped to 'NaN' in the JSON.
    # It represents the class being indeterminable from procedure code and description.
    else:
        return int(len(list(mapping["Internal"]["Code"].keys()))-1)
    
def Resample_Image(img, target_size, inputdim='3D'):
    '''
    Resample images to target size with SimpleITK, target_size must be a tuple, img must be an sitkimage.
    inputdim is restricted to '2D' and '3D'
    '''
    if inputdim=='3D':
        original_spacing = img.GetSpacing()
        original_size = img.GetSize()

        out_size = [target_size[0], target_size[1], target_size[2]]
        out_spacing = [
            (original_size[0] * original_spacing[0] / target_size[0]),
            (original_size[1] * original_spacing[1] / target_size[1]),
            (original_size[2] * original_spacing[2] / target_size[2])]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_size)
        resample.SetOutputDirection(img.GetDirection())
        resample.SetOutputOrigin(img.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(img.GetPixelIDValue())
        resample.SetInterpolator(sitk.sitkBSpline)

        return resample.Execute(img)
    elif inputdim=='2D':
        original_spacing = img.GetSpacing()
        original_size = img.GetSize()

        out_size = [target_size[0], target_size[1], original_size[2]]
        out_spacing = [
            (original_size[0] * original_spacing[0] / target_size[0]),
            (original_size[1] * original_spacing[1] / target_size[1]),
            (original_size[2] * original_spacing[2] / original_size[2])]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_size)
        resample.SetOutputDirection(img.GetDirection())
        resample.SetOutputOrigin(img.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(img.GetPixelIDValue())
        #resample.SetInterpolator()
        return resample.Execute(img)
        
    else:
        raise ValueError('Unsupported dimensionality for resampling! (inputdim must be "3D" or "2D")')
    
def make_representation_from_unknown(current_image, target_size, verbose = False):
    
    lim2d = 40
    
    # Get Dimensionality
    dim = current_image.GetDimension()
    
    # Remove RGB components
    if current_image.GetNumberOfComponentsPerPixel() >= 3:
        if verbose:
            print("Detected vector pixels, # of channels is "+str(current_image.GetNumberOfComponentsPerPixel()))
        channels = [sitk.VectorIndexSelectionCast(current_image,i, sitk.sitkFloat32) for i in range(current_image.GetNumberOfComponentsPerPixel())]
        current_image = sitk.Compose([sitk.Clamp((channels[0]+channels[1]+channels[2])/3., sitk.sitkUInt8)])
        if verbose:
            print("Removed vector pixels, # of channels is now "+str(current_image.GetNumberOfComponentsPerPixel()))
    elif current_image.GetNumberOfComponentsPerPixel() == 2:
        if verbose:
            print("Unknown pixel type (2 channels) received. Skipping series.")
        refuse_garbage_input()
    
    # Resample image, if necessary (If 3D but very flat in Z, do not interpolate, instead treat it as 2D)
    if verbose:
        print(current_image.GetSize())
    if current_image.GetSize() != target_size:
        if dim == 2 or (dim==3 and current_image.GetSize()[2] <= lim2d):
            if verbose:
                print('Resampling 2D (or 3D flat) image ...')
            temp = Resample_Image(img=current_image, target_size=target_size, inputdim='2D')
            if verbose:
                print('Resampled.')
        elif dim ==3 and current_image.GetSize()[2] > lim2d:
            if verbose:
                print('Resampling 3D image ...')
            temp = Resample_Image(img=current_image, target_size=target_size, inputdim='3D')
            if verbose:
                print('Resampled.')
        elif dim ==4 and current_image.GetSize()[2] <= lim2d:
            if verbose:
                print('Resampling flat 3D+CH/T image ...')
            middle_CH_T = int(np.ceil((current_image.GetSize()[3]-1)/2))
            temp = Resample_Image(img=current_image[:,:,:,middle_CH_T], target_size=target_size, inputdim='2D')
            if verbose:
                print('Resampled.')
        elif dim ==4 and current_image.GetSize()[2] > lim2d:
            if verbose:
                print('Resampling 3D+CH/T image ...')
            middle_CH_T = int(np.ceil((current_image.GetSize()[3]-1)/2))
            temp = Resample_Image(img=current_image[:,:,:,middle_CH_T], target_size=target_size, inputdim='3D')
            if verbose:
                print('Resampled.')
        else:
            if verbose:
                print("Number of dimensions ("+str(dim)+") caused error in resampling.")
            refuse_garbage_input()
    if verbose:
        print(temp.GetSize())

    # Construct representation from MIPs of 3D or layers of 2D
    if verbose:
        print('Computing MIP ...')
    temp = sitk.GetArrayFromImage(temp)
    
    if dim == 2:
        s1 = 0
        s2 = np.floor((temp.shape[0]-1)/2).astype(int)
        s3 = -1
        new_image = np.vstack([np.expand_dims(temp[s1,:,:], axis=0),
                               np.expand_dims(temp[s2,:,:], axis=0),
                               np.expand_dims(temp[s3,:,:], axis=0)]).astype(np.float32)
        
    elif dim >= 3 and np.size(temp, -3) <= lim2d:
        s1 = 0
        s2 = np.floor((temp.shape[0]-1)/2).astype(int)
        s3 = -1
        new_image = np.vstack([np.expand_dims(temp[s1,:,:], axis=0),
                               np.expand_dims(temp[s2,:,:], axis=0),
                               np.expand_dims(temp[s3,:,:], axis=0)]).astype(np.float32)
        
    elif dim >= 3 and np.size(temp, -3) > lim2d:
        new_image = np.stack((np.max(temp, axis=-1),
                              np.max(temp, axis=-2),
                              np.max(temp, axis=-3)), axis=0).astype(np.float32)

    if verbose:
        print('Done.')
    return new_image
    
        
def load_single_series(current, path, idx, steps, target_size, mapping, position_in_folder=0, nopet=False, ispi=False):
    # This will load a 3D DICOM image and resample it to target_size, then extract 2D representations with MIP
    try:
        i = idx
        print('Loading data from directory '+str(i+1)+' of '+str(steps))#, end="\r", flush=True)

        # read DICOM header of first file in current dir, extract modality and shape
        meta = pydicom.filereader.dcmread(path+'/'+os.listdir(path)[0], stop_before_pixels=True)

        # get class label from Study Description field in DICOM header
        # This assumes that all series in the folder have the same class. This should be true, but can be false if the
        # study/series was constructed incorrectly. If this was the case, the study would likely break the reader anyway
        try:
            Code = meta[0x0008, 0x1032].value[0][0x0008, 0x0100].value
        except:
            Code = ""
        try:
            Desc = meta[0x0008, 0x1030].value
        except:
            Desc = ""
        try:
            Moda = meta[0x0008, 0x0060].value
        except:
            Moda = ""
        try:
            current_class = get_class(mapping=mapping, c=Code, d=Desc, nopet=nopet, ispi=ispi, m=Moda)
            print(current_class, Code, Desc, Moda)
        except:
            print("Could not find metadata, or get_class failed")
            raise

        # make representation of the series (if more than one series is in the folder, choose one)
        new_image = make_representation_from_unknown(current_image=current[position_in_folder], target_size=target_size)

        # return image, class, metadata
        return (torch.Tensor(new_image), current_class, meta)
        
    except:
        print('Load_single failed for path '+str(path))
        raise
        
def preprocess_training_data(src, dest, target_size, mapfile='./MCMapping.json', cat=0, debug_mode=0, save_meta=False, nopet=False, ispi=False):
    DL = getDirectoryList(src)
    steps = len(DL)
    with open(mapfile) as json_file:
        mapping = js.load(json_file)
    add = 0
    for idx, path in enumerate(DL):
        try:
            current = sorted(sitk_read(path))
            for position in range(len(current)):
                if position != 0:
                    add += 1
                print(str(len(current))+" series found in folder. Preprocessing series #"+str(position+1))
                out_T, out_C, meta = load_single_series(current, path, idx, steps, target_size, mapping, position_in_folder = position, nopet = nopet, ispi = ispi)
                print("Normalizing ...")
                out_T -= out_T.min()
                out_T /= out_T.max()
                if torch.isnan(out_T).any() == False:
                    torch.save((out_T, out_C), dest+'/preprocessed/'+'MM_'+str(int(idx+add+cat)).zfill(8)+'.pth')
                    print("Sample saved. ("+str(int(idx+add+cat))+")")
                    if save_meta:
                        try:
                            meta.save_as(dest+'/metadata/'+'MM_meta_'+str(int(idx+add+cat)).zfill(8)+'.dcm')
                            print("Metadata saved.")
                        except:
                            os.remove(dest+'/metadata/'+'MM_'+str(int(idx+add+cat)).zfill(8)+'.pth')
                            print("Could not save metadata, removed sample from training.")
                else:
                    print("NaN(s) found in output tensor.")
                    refuse_garbage_input()
        except Exception as e:
            print('Preprocessing failed. Excluding series from training data.')
            if debug_mode == 2:
                print('Err: '+str(e))
            if debug_mode == 1:
                raise
        
def clean_training_data(data_root, has_meta=False):
    osl = sorted(os.listdir(data_root))
    tFL = [f for f in osl if f.endswith('.pth')]
    if has_meta:
        mFL = [f for f in osl if f.endswith('.dcm')]
        if len(tFL) != len(mFL):
            print("Warning: # of training images != # of metadata files!")
    for sb in range(len(tFL)):
        os.rename(os.path.join(data_root, tFL[sb]), os.path.join(data_root, 'MM_'+str(sb).zfill(8)+'.pth'))
        if has_meta:
            os.rename(os.path.join(data_root, mFL[sb]), os.path.join(data_root, 'MM_meta_'+str(sb).zfill(8)+'.dcm'))
        
def select_training_data(img_src, meta_src, img_dest, meta_dest, s_map, f_map='./MCMapping.json', keep_pet=True):
    """
    Select training data from img_src with metadata in meta_src and copy it to img_dest/meta_dest, if modality matches.
    If keep_pet==False, anything with modality PT will be ignored while copying.
    s_map must be a valid .json like MCMapping.json and include all relevant training data. This means, for example,
    that to train CT, the CT_only.json will not suffice (these are for the MOMO script and nothing else!)
    """
    with open(f_map) as f_json:
        f_mapping = js.load(f_json)
    with open(s_map) as s_json:
        s_mapping = js.load(s_json)
    
    sb = 0
    tld = sorted(os.listdir(img_src))
    mld = sorted(os.listdir(meta_src))
    tFL = [f for f in tld if f.endswith(".pth")]
    mFL = [f for f in mld if f.endswith(".dcm")]
    
    # Should be auto-sorted because Python 3 is my friend
    dck = list(s_mapping["Internal"]["Desc"].keys())
    dcv = list(s_mapping["Internal"]["Desc"].values())
    
    for i in range(len(tFL)):
        meta_read = pydicom.filereader.dcmread(str(meta_src)+"/"+str(mFL[i]))
        ismod = meta_read["0x0008", "0x0060"].value
        data, f_target = torch.load(str(img_src)+"/"+str(tFL[i]))
        desc = f_mapping["Internal"]["Desc"][str(f_target)]
        if desc in dcv and (keep_pet or ismod != "PT"):
            s_target = dck[dcv.index(desc)]
            isp = str(img_src)+"/"+str(tFL[i])
            idp = str(img_dest)+"/"+"MM_"+str(sb).zfill(8)+".pth"
            msp = str(meta_src)+"/"+str(mFL[i])
            mdp = str(meta_dest)+"/"+"MM_meta_"+str(sb).zfill(8)+".dcm"
            torch.save((data, s_target), idp)
            shutil.copy(src=msp, dst=mdp)
            sb += 1
    return None

class LAZY_Dataset(torch.utils.data.Dataset):
    """
    Dataset generator class. Lazily reads samples from disk as __getitem__ is called by the DataLoader during
    batch creation. This is slower than storing all data in RAM, but often the only choice, if the dataset
    exceeds your memory budget. Consequently, the data must first be fed and then the dataset can
    be created.
    
    data_root specifies the directory in which the data is recursively searched for. Must be valid path.
    
    __getitem__ returns the image tensor, image class and dataset index
    
    __getmeta__ returns the dcm metadata and index
    """
    
    def __init__(self,                  #
                 data_root,             # root folder with training data
                 p_rotflip,             # probability for rotation/flip transforms of data in __getitem__
                 custom_transform=None, # custom transformation to apply to all input images
                 for_training=True):    # whether the dataset (or the part of it, if split later) is for training or testing

        # grab all paths
        self.FL = [f for f in sorted(os.listdir(data_root+'preprocessed/')) if f.endswith('.pth')]
        self.steps = len(self.FL)
        self.data_root = data_root
        self.p = p_rotflip
        self.training = for_training
        self.custom_transform = custom_transform

    def __len__(self):
        return self.steps
    
    def __getitem__(self, idx):
        # load tensor from directory
        data, target = torch.load(self.data_root+'preprocessed/MM_'+str(idx).zfill(8)+'.pth')
        
        # apply random transformations
        if self.custom_transform:
            data = self.custom_transform(data)
        
        if self.training:
            rn = np.random.uniform(0,1)
            if self.p < rn:
                deg = 0
            else:
                deg = 90 * np.random.randint(1,4)
                data = torchvision.transforms.functional.rotate(data, deg)
            flip = transforms.Compose([torchvision.transforms.RandomHorizontalFlip(p=self.p)])
            data = flip(data)
        
        # return item
        return data, int(target), idx
    
    def __getmeta__(self, idx):
        meta = pydicom.dcmread(self.data_root+'/metadata/MM_meta_'+str(idx).zfill(8)+'.dcm')
        return meta, idx