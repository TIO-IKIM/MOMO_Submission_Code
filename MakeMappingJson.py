from argparse import ArgumentParser
import os
import json
import pandas as pd
import numpy as np

# Make a JSON from and ods or xlsx file.

parser = ArgumentParser()
parser.add_argument("-i", "--infile", dest="infile", default="./Procedures.xlsx", help="Use as -i /path/to/procedure/table.xlsx")
parser.add_argument("-o", "--outfile", dest="outfile", default="./MCMapping.json", help="Use as -o /path/to/outjson")
parser.add_argument("-m", "--multiclass", dest="multiclass", default=None, help="If you want to add procedure codes that get mapped to different ones, add them like this: -m '{'KHAND': ['KHANDL', 'KHANDR']}'. It maps right and left hand procedure codes to one common class for hands. Note: If python complains, use double quotes inside the curly brackets.")
parser.add_argument("-p", "--petmap", dest="petmap", default=None, help="If you want to add procedure codes which have a PET equivalent, such as when PET-CT studies get evaluated by the neural network for CTs, this is the place. If you wanted to map a CT Skull (CTS) to PET CT Skull (PCTSC), you would add it like this: -p '{'CTS': 'PCTSC'}'. Note: If python complains, use double quotes inside the curly brackets.")
args = parser.parse_args()
src = args.infile
dest = args.outfile
multiclass = json.loads(args.multiclass)
petmap = json.loads(args.petmap)

mdl = {"Internal":
    {"Code":{},
     "Desc":{},
     "Merr":{},
     "Moda":{},
     "Multiclass":{},
     "Alts":{}
     },
       "External":
            {"Code":{},
             "Desc":{},
             "Moda":
                {"CT":{},
                 "XA":{},
                 "CR":{},
                 "MG":{},
                 "MR":{},
                 "PT":{},
                 "US":{}
                 }
               },
       "PETMap":{}
       }

if src.endswith(".ods"):
    frame = pd.read_excel(src, engine="odf")
elif src.endswith(".xlsx"):
    frame = pd.read_excel(src)
else:
    raise TypeError("Infile must be of type .ods or .xlsx")

for idx, key in enumerate(frame["Procedure Code"]):
    mdl["Internal"]["Code"][idx] = frame["Procedure Code"][idx]
    mdl["Internal"]["Desc"][idx] = frame["Full Name"][idx]
    if frame["Minor Error"][idx] == frame["Minor Error"][idx]:
        mdl["Internal"]["Merr"][idx] = frame["Minor Error"][idx]
    else:
        mdl["Internal"]["Merr"][idx] = ""
    mdl["Internal"]["Moda"][idx] = frame["Modality"][idx]
    mdl["Internal"]["Alts"][idx] = frame["Alternatives"][idx]
mdl["Internal"]["Code"][idx+1] = np.nan
mdl["Internal"]["Desc"][idx+1] = np.nan
mdl["Internal"]["Merr"][idx+1] = np.nan
mdl["Internal"]["Moda"][idx+1] = np.nan
mdl["Internal"]["Alts"][idx+1] = np.nan

for key in sorted(list(multiclass.keys())):
    for item in multiclass[str(key)]:
        mdl["Internal"]["Multiclass"][str(item)] = str(key)
for key in sorted(list(petmap.keys())):
    mdl["PETMap"][str(key)] = petmap[str(key)]

print("Submitted "+str(idx+1)+" classes to JSON.")

with open(dest, 'w') as out:
    json.dump(mdl, out)
