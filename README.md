The project code is meant to be open source, which means you can run it, keep it, copy it, modify it, etc. as long as you
mention my name and paper somewhere and point to the original. Have fun.
This should go without saying, but this project is a WIP and has had limited testing. If you implement it somewhere where
its failure can cause damage, please pay close attention to what it does and whether it works properly - If a bug has
escaped me, I will attempt to fix it.

Installation instructions:

1) Install the environment. This can be done using anaconda/miniconda: 'conda env create --file env_MEDDL.yml'
 All scripts require being run in the conda environment. Activate it like this: 'conda activate MEDDL'.
2) Download the networks from [https://cloud.uk-essen.de/d/9419e258dce34cabb848/] and put them into the main folder.
3) Run 'python DownloadSample.py' (If there is an error with 7z not being installed, you have to unpack it yourself)
4) Run 'python MOMO.py -v -s ./imports/sample/'

NOTE: I compiled this entire project using miniconda3 on Ubuntu 20.04 LTS. The project should work on other
 systems aswell, since everything happens inside a conda environment, but I have not extensively tested that.
 
NOTE: The entire project was run on a Black Titan GPU from NVIDIA. In theory it will work with any graphics
 card which is supported by the pytorch version supplied in the environment, but I have only tested it on
 a GTX 1080. If you have no or no useful GPU, you have to edit any script which uses a GPU to use the CPU
 instead (TransferDense.ipynb and TranserRes.ipynb). Find the line starting with 'device = ...' and edit
 it so it either checks if cuda is available on your system, or even edit it so it only uses the CPU.
 With a GPU, the runtime of the program should be at most 10 seconds for a series, depending on the size
 and GPU, with CPU the runtime is much higher (probably around 30-60s, I have not tested extensively).

Execution instructions and notes:

*_config.ini - The config file which is read during execution of MOMO. If you want to experiment with
 MOMO, copy a file, edit it to your heart's content (explanations on how to do that are in the file), and
 then run MOMO using that config file. The default_config.ini was used to evaluate all algorithm variants
 except algo=12, for which the rr_config.ini was used for technical reasons. 

*.json - If you only want to test this program, don't touch these. These files contain the known procedure
 codes, study descriptions, minor error definitions and associated keywords for all study classes.

DatasetFunctions.py - A list of utility functions I used when assembling the dataset and wrangling pytorch.
 You shouldn't need to touch this.
 
DownloadSample.py - Run as 'python DownloadSample.py'. This will access a google drive containing some
 sample DICOM file, which ends up in ./imports/ .
 If you do not have 7zip installed, there will be a shell console complaint. In this case, manually
 extract the .7z archive inside that folder.
 The sample file is a single-series study, the series is an MRI of the head region. Notably, the study
 description is completely missing in the study, but series metadata and machine-learning image analysis
 both recognize it as the head region.
 
 The sample file is a public sample provided by https://www.aliza-dicom-viewer.com/download/datasets
 (Siemens, diffusion/trace, 3D+Bvalue), as we cannot upload any of our patient data as publicly available
 sample files.
 
MakeMappingJson.py - Run as 'python MakeMappingJson.py -i infile -o outfile -m [multiclass instructions] -p [petmap instructions]'.
 'infile' is the tabular document from which you generate the mapping
 'outfile' is the destination of json you create
 'multiclass' maps multiple elements onto one class. Some hospitals will not care if an X-ray of the left or
  right hand has come in, only that it was an X-ray of the hand. Format the input like this: -m '{"KHAND": ["KHANDL", "KHANDR"]}' 
 'petmap' maps a number of classes to precisely one other class, a PET class. This is because we don't actually
  analyze PET images with the neural networks, but rather the CT or MRI components of PET studies. The network
  for CT images figures out the body region and then picks the corresponding PET class. If you want to map
  CT Skull (CTS) studies to PET CT Skull (PCTSC) studies, do it like this: -p '{"CTS": "PCTSC"}'
  
MOMO.py - The main function. Mostly just wraps stuff from MOMO_Backbone.py. Execute this function like:
 'python MOMO.py -s ./path/to/study/folder'
 There is additional flags explained by the argparser. Note that -s expects the location of ONE study folder
 and not a folder containing a bunch of studies! You will need to make an extra two lines of wrapper for that.
 If you downloaded the sample DICOM image, you can run 'python MOMO.py -v -s ./imports/sample/' .
 The result should be a lot of text (the metadata reading and matching) and a final result, consisting of
 exitcode, result, decider, and potential error. The result should be '0, ("MRS", Merged Voting), None'.

MOMO_Backbone.py - Function definitions for MOMO. If you want to experiment or go through code, this is
 where you do it. If you only want to test things or use the program, don't touch this.
 
MonteCarloExperiment.py - Execute as 'python MonteCarloExperiment.py'. This will run the experiment as
 described in the paper and deposit the output plot in the current folder.
 
Radiology_Procedures.xlsx - This is the classes which our hospital uses and which we try to map all external
 studies to. We also created the labels for our testset using these classes. This is the sort of document which
 MakeMappingJson.py takes as input. If you want to make your own version of MOMO, you should create a table
 like this and run it through MakeMappingJson.py (aswell as make your own neural networks). Note that the
 document is the original document used for our evaluation and as such covers German and English phrases as
 the two most common.
 
README.md - The best damn readme in the world :v)

Transfer*.ipynb - The notebooks in which the function definitions for the networks can be found. TransferNet_script.py
 is an example of how you can make and structure a custom script which MOMO can import without needing modifications.
 You can copy these and play around with them to make your own networks.
 (They are still in a notebook format to make it so you can explore and play around with them more easily.)
