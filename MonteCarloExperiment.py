import os
import time

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import random
from scipy import stats

from collections import Counter
from tqdm import tqdm

def prediction_metastudy(n_opts, n_per_study=5, n_studies=100, p_correct=0.7, rnosps=True, worstcase=False):
    options = [str(i) for i in range(n_opts)]
    snoitpo = options[::-1]
    if rnosps: # random number of series per study
        truths = [(options[int(np.floor(np.random.uniform(0,n_opts)))], max(np.around(np.random.randn(1)[0]*2 + n_per_study),1)) for x in range(n_studies)]
    else:
        truths = [(options[int(np.floor(np.random.uniform(0,n_opts)))], n_per_study) for x in range(n_studies)]
    predictions = []
    std = (1-p_correct)/2 # p to 1 is 2sigma
    for truth, N in truths:
        votes = []
        for n in range(int(N)):
            # roll beta distributed W with a+b=4, a/(a+b)=p_correct
            # this approximately simulates network output, where on average n_correct/n_all = p_correct
            # then roll c uniform in [0,1] and make it a correct prediction if c<w
            # this satisfies the calibration condition we assume our network also satisfies
            w = np.random.beta(4*p_correct, 4-(4*p_correct))
            #if p_correct == 0.5:
            #    w = 0.5
            c = np.random.uniform(0,1)
            if c < w:
                votes.append((truth, w))
            else:
                while True:
                    if worstcase:
                        trash = snoitpo[options.index(truth)] # maximally correlated mispredictions
                    else:
                        trash = options[int(np.floor(np.random.uniform(0,n_opts)))] # uncorrelated mispredictions
                    if trash != truth:
                        break
                    else:
                        if worstcase:
                            print("This should never be reached")
                votes.append((trash, w))
        counter = {}
        for vote in votes:
            if vote[0] in list(counter.keys()):
                counter[vote[0]] += vote[1]
            else:
                counter[vote[0]] = vote[1]
        mv = max(counter.values())
        winners = [key for i, key in enumerate(list(counter.keys())) if list(counter.values())[i] == mv]
        predictions.append(winners[0])
    acc = sum([1 if p==truths[i][0] else 0 for i, p in enumerate(predictions)])/len(truths)
    return acc

n_pers = [1,2,3,4,5,6,7,8,9,10]
p_cors = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
bsi = 10 # bootstrapping instances

UC_means = np.zeros((len(n_pers), len(p_cors)))
UC_stds = np.zeros((len(n_pers), len(p_cors)))
MC_means = np.zeros((len(n_pers), len(p_cors)))
MC_stds = np.zeros((len(n_pers), len(p_cors)))

for i, n_per in enumerate(tqdm(n_pers)):
    for j, p_cor in enumerate(p_cors):
        outs = []
        for k in range(bsi):
            out = prediction_metastudy(n_opts = 12,
                                       n_per_study=n_per,
                                       n_studies=1500,
                                       p_correct=p_cor,
                                       rnosps = False,
                                       worstcase = False)
            outs.append(out)
        UC_means[i,j] = np.mean(outs)
        UC_stds[i,j] = np.std(outs) # /np.sqrt(bsi)
for i, n_per in enumerate(tqdm(n_pers)):
    for j, p_cor in enumerate(p_cors):
        outs = []
        for k in range(bsi):
            out = prediction_metastudy(n_opts = 12,
                                       n_per_study=n_per,
                                       n_studies=1500,
                                       p_correct=p_cor,
                                       rnosps = False,
                                       worstcase = True)
            outs.append(out)
        MC_means[i,j] = np.mean(outs)
        MC_stds[i,j] = np.std(outs) # /np.sqrt(bsi)
        
# error plot with legend, what performance should i be expecting for study predictions, given a network performance
# dashed lines mark maximally correlated (worst case) mispredictions scenarios, which should be our minimum
# accuracy on any real dataset

cs = plt.cm.get_cmap('tab10', len(n_pers))

plt.figure(figsize=(16,9))
for i in range(len(n_pers)):
    plt.errorbar(x=p_cors, y=UC_means[i,:], xerr=None, yerr=UC_stds[i,:], label=str(n_pers[i])+" series per study", color=cs(i), capsize=3)
    plt.errorbar(x=p_cors, y=MC_means[i,:], xerr=None, yerr=MC_stds[i,:], color=cs(i), linestyle="--", capsize=3)
plt.grid(b=True)
plt.title("Simulated study prediction accuracy (bootstrapped)", fontsize=20)
plt.legend(loc="lower right", markerscale=2.5, fontsize=15)
plt.xlabel("Network accuracy (per series)", size=15)
plt.ylabel("Study prediction accuracy", size=15)
plt.xlim(0.48, 1)
plt.ylim(0.45, 1.05)
plt.xscale("linear")
plt.yscale("linear")
plt.savefig("./MonteCarloExperiment.png")

# What is my average n_per_study in the imported external data? => 9.291056910569106
# Note that this last bit will naturally not work without having acquired the data used in the study

blacklist = ["patient", "protocol", "dosis", "report", "topogram", "scout", "screen", "save", "scoring", "evidence", "document", "result", "text"]

PIF = "./imports"

studydirs = [PIF+d for d in os.listdir(PIF) if os.path.isdir(PIF+d)]
lengths = []
for p in studydirs:
    x = [1 for series in os.listdir(p) if (os.path.isdir(p+"/"+series) and not any(substring in series for substring in blacklist))]
    lengths.append(len(x))
print(np.mean(lengths))
