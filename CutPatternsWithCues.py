from __future__ import division
import os
import fnmatch
import re
import matplotlib.pyplot as plt
import numpy
from operator import itemgetter
import random
import numpy as np
from fractions import Fraction


path="C:/Users/admin_local/Dropbox/2017Pattern/MeredithTLF1MIREX2016/"
pathout = "C:/Users/admin_local/Dropbox/2017Pattern/MeredithTLF1MIREX2016/cut"
cuepath = "C:/Users/admin_local/Dropbox/2017Pattern/cues"

for root, dirs, files in os.walk(path):
    # files[0:4] for debugging purpose (only the set of the first song)
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

            CueT_1 = 0
            # extracting the cues
            for i, CRaw in enumerate(CurrentCue):
                print('--------------------------------')
                if i == 0:
                    continue
                CueI=CRaw.split('\t')[0]
                Filename = CRaw.split('\t')[1]
                if i == 1:
                    CueT_1 = 0
                else:
                    CueT_1 = CueT

                try:
                    CueT=float(CueI)
                except:
                    CueT=float(Fraction(CueI))

                # value comparison
                flag = True
                for pair in pairs:
                    if pair[0] >= CueT and flag:
                        textfile=open("{0}.txt".format(Filename[0:-4]),"w")
                        flag = False
                        textfile.write(str(pair[0]))
                    else:
                        print("text")
                        print(pair[0])
                        print(CueT)
                        print(CueT_1)

                        if pair[0] >= CueT and pair[0] <= CueT_1:
                            print(CueT)
                            print(CueT_1)
                            textfile.write(str(pair[0]))
                        else:
                            break
                            # print(pair[0])
                            # print(CueT)
                            # print(CueT_1)
                            # print(Filename[0:-5])


                # if pair[0] < CurrentCue[index]:
                #     print(True)
