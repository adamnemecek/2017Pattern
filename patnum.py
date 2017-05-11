from __future__ import division
import os
import fnmatch
import re

def outputtimes(text):
        pitches=[]
        pairs=[]
        occurtimes=[]
        pattimes=[]
        times=[]
        patternnum=[]
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
                patternnum.append(re.findall('\d+', line))
                total.append('p')
                pattimes.append(occurtimes)
                # print(len(occurtimes))
                occurtimes=[]

        # print('good'+str(pattimes))
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
        return startendpat, patternnum

def outputtimeandpitch(text):
    pitches=[]
    pairs=[]
    occurtimes=[]
    occurpitches=[]
    patpitches=[]
    pattimes=[]
    times=[]
    total=[]
    index=0
    for line in text:
        if "," in line:
            pairs.append([float(i) for i in line.split(',')])
            total.append([float(i) for i in line.split(',')])

        if "o" in line:
            total.append('o')
            if pairs != []:
                times=zip(*pairs)[0]
                # print(times)
                pitches=zip(*pairs)[1]
                occurtimes.append(times)
                occurpitches.append(pitches)
                pairs=[]

        if "p" in line:
            total.append('p')
            pattimes.append(occurtimes)
            patpitches.append(occurpitches)
            # print(len(occurtimes))
            occurtimes=[]
            occurpitches=[]
        index=index+1

    # print(total)
    # print(pattimes)
    # print(patpitches)

    olist=[]
    plist=[]
    for index in range(0,len(total)):
        item = total[index]
        if item == 'p':
            plist.append(index)
        if item =='o':
            olist.append(index)

    occurtimes=[]
    occurpitches=[]
    pattimes=[]
    patpitches=[]
    record=0

    for pindex in range(1,len(plist)):
        for oindex in range(0,len(olist)-1):
            # print('pindex='+str(plist[pindex])+';oindex='+str(olist[oindex+1]))
            # print(zip(*total[olist[oindex]+1:olist[oindex+1]]))
            # print(zip(*total[olist[oindex]+1:olist[oindex+1]-1]))
            if plist[pindex]-olist[oindex+1]>1 and oindex>=record:
                # print('first'+str(oindex)+str(pindex))
                occurtimes.append(zip(*total[olist[oindex]+1:olist[oindex+1]])[0])
                occurpitches.append(zip(*total[olist[oindex]+1:olist[oindex+1]])[1])
            if plist[pindex]-olist[oindex+1]==-1:
                # print('second'+str(pindex)+str(pindex))
                occurtimes.append(zip(*total[olist[oindex]+1:olist[oindex+1]-1])[0])
                occurpitches.append(zip(*total[olist[oindex]+1:olist[oindex+1]-1])[1])
                record=oindex+1
                # print(record)
        sub=[]
        # print(occurtimes)
        # print(occurpitches)
        
        pattimes.append(occurtimes)
        patpitches.append(occurpitches)
        occurtimes=[]
        occurpitches=[]
    # print(olist)
    # print(plist)
    # print(patpitches)
    pindex=plist[-1]
    occurtimes=[]
    occurpitches=[]
    for oindex in range(0,len(olist)-1):
        if olist[oindex]>pindex:
            occurtimes.append(zip(*total[olist[oindex]+1:olist[oindex+1]-1])[0])
            occurpitches.append(zip(*total[olist[oindex]+1:olist[oindex+1]-1])[1])


    oindex=olist[-1]
    occurtimes.append(zip(*total[oindex+1:])[0])
    occurpitches.append(zip(*total[oindex+1:])[1])
    pattimes.append(occurtimes)
    patpitches.append(occurpitches)

    # print(patpitches)
    # print(pattimes)
    # taking the onset and offset
    startend=[]
    startendpitches=[]
    startendpat=[]
    startendtime=[]
    startendpatpitches=[]
    startendpattime=[]
    # print(pattimes)
    for occtimei in range(0,len(pattimes)):
        occtime = pattimes[occtimei]
        occpitch= patpitches[occtimei]
        # print(occtime)
        # print(occpitch)

        for timei in range(0,len(occtime)):
            start=occtime[timei][0]
            end=occtime[timei][-1]
            startend.append([start,end])
            startendtime.append(occtime[timei])
            startendpitches.append(occpitch[timei])

        startendpat.append(startend)
        startendpatpitches.append(startendpitches)
        startendpattime.append(startendtime)
        startend=[]
        startendtime=[]
        startendpitches=[]

    # print(startendpat[-1]) 
    return startendpat, startendpatpitches,startendpattime

f={}
m=0
c=0
prefamily=0
memory=[]
familyindex=[]
for files in os.listdir("C:/Users/admin_local/Desktop/filesboot/NLB/AnnotatedMotifs/discovery/"):
    address=os.path.join("C:/Users/admin_local/Desktop/filesboot/NLB/AnnotatedMotifs/discovery", files)
    
    if fnmatch.fnmatch(files, "*.txt"):
        f[m] = open(address, 'r')
        m += 1

        for s in files:
            if s == '+':
                break
            memory.append(s)
        family=''.join(memory)

        if m != 0:
            if family != prefamily:
                c += 1

        prefamily=family
        memory=[]
        familyindex.append(c)

print(familyindex)
d={}
startendpat={}
startendpatpitches={}
patternnums={}
startendpattime={}
for i in range(0,m-1):
    d[i]=f[i].readlines()
    if d[i] != []:
        # output=outputtimes(d[i])
        output=outputtimeandpitch(d[i])
        startendpat[i]=output[0]
        patternnums[i]=outputtimes(d[i])[1]
        startendpatpitches[i]=output[1]
        startendpattime[i]=output[2]
        # startendpatpitches[i]=output[1]
    else:
        startendpat[i]=[]
        # startendpatpitches[i]=[]
        patternnums[i]=[]

# print(len(startendpat))
import matplotlib.pyplot as plt
import numpy
height = 0
from operator import itemgetter

import random
import numpy as np
extraheight=0
for tunenum in range(1,26):
    plt.figure(figsize=(10,15))
    for i in range(0, m-1):
        if familyindex[i]==tunenum:
            c=numpy.random.rand(3,1)
            for color in c:
                if color < 0.5:
                    color = color + 0.5
            
            for n in range(0,len(startendpat[i])):
                patterns = startendpat[i][n]
                t=startendpattime[i][n]
                p=startendpatpitches[i][n]

                extraheight = extraheight + 0.01
                height = int(patternnums[i][n][0])+extraheight
                for j in range(0, len(patterns)):
                    occur = patterns[j]
                    tcur = t[j]
                    pcur = p[j]

                    # plt.plot((occur[0], occur[1]), (height, height), color = c, lw=2, alpha=0.5)
    #                 plt.scatter(tcur, [pitch * 0.008 + height for pitch in pcur], color = c, alpha=0.5, s = 4)
    #                 plt.xlabel('Time')
    #                 plt.yticks([])
    #                 # plt.title('The location of music patterns in a tune family')
    #                 plt.title('The location and pitch of music patterns in a tune family')
    # plt.savefig('dutchmelo'+str(tunenum))
# plt.show()

import csv
import scipy.stats
with open('features.csv', 'w') as file:
    # spamwriter = csv.writer(csvfile, delimiter=',')
    fields = ('start', 'end','relastart','relaend','startnote','endnote','intervalsize','entropy','class')
    wr = csv.DictWriter(file, fieldnames=fields, lineterminator = '\n')
    wr.writeheader()
    
    # spamwriter.writerow(['start']+['end']+['class'])

    for tunenum in range(1,5):
        for i in range(0, m-1):
            if familyindex[i]==tunenum:         
                for n in range(0,len(startendpat[i])):
                    patterns = startendpat[i][n]
                    t=startendpattime[i][n]
                    tmax = max(max(max(startendpat[i])))
                    print(tmax)
                    p=startendpatpitches[i][n]
                    interval=[]
                    for j in range(0, len(patterns)):
                        occur = patterns[j]
                        tcur = t[j]
                        pcur = p[j]
                        for pitchindex in range(0,len(pcur)-1):
                            interval.append(pcur[pitchindex+1]-pcur[pitchindex])
                        intervalsize = len(set(interval))
                        entropy = scipy.stats.entropy(p[j])
                        # spamwriter.writerow([occur[0]]+[occur[1]]+[tunenum])
                        # wr.writerow({'start':occur[0], 'end': occur[1],'relastart':occur[0]/tmax,'relaend':occur[1]/tmax,'startnote':pcur[0],'endnote':pcur[-1],'intervalsize':intervalsize, 'entropy':entropy,'class':str(tunenum)+'tune'})
                        wr.writerow({'start':occur[0], 'end': occur[1],'relastart':random.uniform(0, 1),'relaend':random.uniform(0, 1),'startnote':pcur[0],'endnote':pcur[-1],'intervalsize':intervalsize, 'entropy':entropy,'class':str(tunenum)+'tune'})