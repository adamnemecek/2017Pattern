import os
import fnmatch
from fractions import Fraction
import matplotlib.pyplot as plt

PathCue = "C:/Users/admin_local/Dropbox/2017Pattern/cues/Daar_ging_een_heer_1.txt"
CueFile = open(PathCue,'r')
Cues = []
for i, CRaw in enumerate(CueFile):
    CueI=CRaw.split('\t')[0]
    Filename = CRaw.split('\t')[1]

    try:
        CueT=float(CueI)
    except:
        CueT=float(Fraction(CueI))

    Cues.append(CueT)

PathAnno = "C:/Users/admin_local/Desktop/filesboot/nlb/AnnotatedMotifs/discovery"

# Plot ground truth in sparate files

# Plot ground truth in concatenate version

# Plot discovered patterns in sparate files
singleFpath =  "C:/Users/admin_local/Desktop/filesboot/nlb/AnnotatedMotifs/discovery/Daar_ging_een_heer_1+NLB072587_01.txt"

# Plot discovered patterns in concatenate version
path="C:/Users/admin_local/Dropbox/2017Pattern/MeredithTLF1MIREX2016/Daar_ging_een_heer_1.txt.tlp"

def outputtimes(text):
    pitches=[]
    pairs=[]
    occurtimes=[]
    pattimes=[]
    times=[]
    total=[]
    for line in text:
        if "," in line:
            pairs.append([float(i) for i in line.split(',')])
            total.append([float(i) for i in line.split(',')])

        if "o" in line:
            total.append('o')
            if pairs != []:
                times=zip(*pairs)[0]
                occurtimes.append(times)
                pairs=[]

        if "p" in line:
            total.append('p')
            pattimes.append(occurtimes)
            # print(len(occurtimes))
            occurtimes=[]

    # print(total)

    olist=[]
    plist=[]
    for index in range(0,len(total)):
        item = total[index]
        if item == 'p':
            plist.append(index)
        if item =='o':
            olist.append(index)

    occurtimes=[]
    pattimes=[]
    record=0
    for pindex in range(1,len(plist)):
        for oindex in range(0,len(olist)-1):
            if plist[pindex]-olist[oindex+1]>1 and oindex>=record:
                occurtimes.append(zip(*total[olist[oindex]+1:olist[oindex+1]])[0])
            if plist[pindex]-olist[oindex+1]==-1:
                occurtimes.append(zip(*total[olist[oindex]+1:olist[oindex+1]-1])[0])
                record=oindex+1
                # print(record)
        sub=[]
        pattimes.append(occurtimes)
        occurtimes=[]

    # print(olist)
    # print(plist)

    pindex=plist[-1]
    occurtimes=[]
    for oindex in range(0,len(olist)-1):
        if olist[oindex]>pindex:
            occurtimes.append(zip(*total[olist[oindex]+1:olist[-1]])[0])


    oindex=olist[-1]
    occurtimes.append(zip(*total[oindex+1:])[0])
    pattimes.append(occurtimes)

    # print(pattimes[1])
    # taking the onset and offset
    startend=[]
    startendpat=[]
    for occtime in pattimes:
        for time in occtime:
            start=time[0]
            end=time[-1]
            startend.append([start,end])
        startendpat.append(startend)
        startend=[]
    # print(startendpat[-1])
    return startendpat

startendpat = outputtimes(open(path,'r').readlines())

import matplotlib.pyplot as plt
import numpy
import scipy.stats as ss


plt.figure()
height=5
for patterns in startendpat:
    # c=numpy.random.rand(3,1)
    height = height + 10
    for occur in patterns:
        # print(occur)
        plt.plot((occur[0], occur[1]), (height, height), color = 'red', lw=2, alpha=0.5)
plt.plot((0,0), (0,0), color='white', label="GT")
for cue in Cues:
    plt.axvline(cue)
plt.ylabel('Pattern Number & Ground Truth Patterns')
plt.xlabel('Time')
# plt.title('The polling curve')
plt.tight_layout()
plt.show()

flag = True
plt.figure()
height=5
valid=[]
for patterns in startendpat:
    # c=numpy.random.rand(3,1)
    height = height + 10
    for occur in patterns:
        for cue in Cues:
            # check all the occurrences against the cues, if there's a violation, move to the next occurrences, preserving at the occurrence level, not at the pattern level
            if (occur[0]<cue and occur[1]>cue):
                flag = False
                break
            else:
                flag = True
                # print(occur)
        if flag:
            plt.plot((occur[0], occur[1]), (height, height), color = 'red', lw=2, alpha=0.5)
            valid.append((occur[0], occur[1]))
plt.plot((0,0), (0,0), color='white', label="GT")
for cue in Cues:
    plt.axvline(cue)
plt.ylabel('Pattern Number & Ground Truth Patterns')
plt.xlabel('Time')
# plt.title('The polling curve')
plt.tight_layout()
plt.show()


flattened_list=[y for x in startendpat for y in x]
startflat=zip(*flattened_list)[0]
endflat=zip(*flattened_list)[1]
totallist=[]
totallist=totallist+flattened_list
totalstartlist=[]
totalstartlist=totalstartlist+list(startflat)
totalendlist=[]
totalendlist=totalendlist+list(endflat)

dist={}
print(totalendlist)
print(startflat)

print(endflat)
dist={}
for time in [x * 1 for x in range(0, int(max(totalendlist)))]:
    dist[time] = 0
    for j in range(0,len(startflat)):
        if time >= startflat[j] and time <= endflat[j]:
            dist[time] = dist[time] + 1

print(sorted(dist.keys()))
plt.figure()
plt.scatter(dist.keys(), dist.values(),s=0.1)
plt.show()

dist={}
for time in [x * 0.5 for x in range(0, int(max(totalendlist)*2))]:
    dist[time] = 0
    for j in range(0,len(startflat)):
        if time >= startflat[j] and time <= endflat[j]:
            dist[time] = dist[time] + 1

print(sorted(dist.keys()))
plt.figure()
plt.scatter(dist.keys(), dist.values(),s=0.1)
plt.show()

for time in [x * 0.25 for x in range(0, int(max(totalendlist)*4))]:
    dist[time] = 0
    for j in range(0,len(startflat)):
        if time >= startflat[j] and time <= endflat[j]:
            dist[time] = dist[time] + 1

print(sorted(dist.keys()))
plt.figure()
plt.scatter(dist.keys(), dist.values(),s=0.1)
plt.show()

dist={}
for time in [x * 1 for x in range(0, int(max(totalendlist)))]:
    dist[time] = 0
    for j in range(0,len(valid)):
        if time >= valid[j][0] and time <= valid[j][1]:
            dist[time] = dist[time] + 1

print(sorted(dist.keys()))
plt.figure()
plt.scatter(dist.keys(), dist.values(),s=0.1)
plt.show()

# for root, dirs, files in os.walk(path):
#     for CurrentFileName in files[0:4]:
#         address= os.path.join(root, CurrentFileName)
#         if fnmatch.fnmatch(CurrentFileName, "*.tlr"):
#             with open(address, 'r') as CurrentFile:
#                 ToBePlot = CurrentFile.readlines()
