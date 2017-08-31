from __future__ import division
import os
import fnmatch
import re
import matplotlib.pyplot as plt
import numpy

from operator import itemgetter

import random
import numpy as np

path="C:/Users/admin_local/Dropbox/2017Pattern/MeredithTLF1MIREX2016/"
pathout = "C:/Users/admin_local/Dropbox/2017Pattern/MeredithTLF1MIREX2016/cut"
cuepath = "C:/Users/admin_local/Dropbox/2017Pattern/cues"

for root, dirs, files in os.walk(path):
    for CurrentFileName in files[0:4]:
        address= os.path.join(root, CurrentFileName)
        if fnmatch.fnmatch(CurrentFileName, "*.tlr"):
            with open(address, 'r') as CurrentFile:
                ToBeCut = CurrentFile.readlines()

            # temp vairable for recording the family name
            memory=[]
            for CheckName in CurrentFileName:
                if CheckName == '+':
                    break
                else:
                    memory.append(CheckName)
            CueFileName=''.join(memory)

            # Now we have the CueFileName from the ToBeCut file, we load the cue file:
            CurrentCueFileAddress = os.path.join("C:/Users/admin_local/Dropbox/2017Pattern/cues", CueFileName)

            # -4 comes from the fact that we named the file "*.txt.tlr"
            with open(CurrentCueFileAddress[:-4], 'r') as CurrentCueFile:
                CurrentCue = CurrentCueFile.readlines()
            print(CurrentCue)
            
            # get the actual pattern data from the ToBeCut file
            pairs=[]
            for line in ToBeCut:
                if "," in line:
                    pairs.append([float(i.replace(',', '.')) for i in line.split(', ')])
            print(pairs)

            # value comparison
            index = 0
            for pair in pairs:
                print(pair[0])
                print(CurrentCue[index])
                index += 1
                # if pair[0] < CurrentCue[index]:
                #     print(True)
            

            