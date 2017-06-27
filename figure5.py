startindex=0
endindex=11
# read files to a nice format
# f={}
# f[0] = open("ref/chopin-mono.txt",'r')
# f[1] = open("2014ON/mazurka24-4-polyoutput.txt",'r')
# f[2] = open("2016MP/chopinOp24No4.txt",'r')
# f[3] = open("OL1/mazurkaoutput.txt",'r')
# f[4] = open("OL2/mazurkaoutput.txt",'r')
# f[5] = open("MeredithTLF1MIREX2016/mazurka24-4output.txt",'r')
# f[6] = open("MeredithTLPMIREX2016/mazurka24-4output.txt",'r')
# f[7] = open("MeredithTLRMIREX2016/mazurka24-4output.txt",'r')
# f[8] = open("VM/VM1/output/mazurka24-4.txt",'r')
# f[9] = open("VM/VM2/output/mazurka24-4input.txt",'r')
# f[10] = open("2016IR/chopin-mono.txt",'r')

f={}
f[0] = open("ref/chopin-mono.txt",'r')
f[1] = open("2014ON/mazurka24-4-polyoutput.txt",'r')
f[2] = open("2016MP/chopinOp24No4.txt",'r')
f[8] = open("OL1/chopin1.txt",'r')
f[9] = open("OL2/chopin2.txt",'r')
f[3] = open("MeredithTLF1MIREX2016/mazurka24-4mono.tlf1",'r')
f[4] = open("MeredithTLPMIREX2016/mazurka24-4.tlf1",'r')
f[5] = open("MeredithTLRMIREX2016/mazurka24-4.tlf1",'r')
f[6] = open("VM/VM1/output/mazurka24-4.txt",'r')
f[7] = open("VM/VM2/output/mazurka24-4input.txt",'r')
f[10] = open("SIARCT-CFP/examples/exampleData/patterns_mazurka24-4.txt")

# f={}
# f[0] = open("ref/bach-mono.txt",'r')
# f[1] = open("2014ON/bach.txt",'r')
# f[2] = open("2016MP/bachBWV889Fg.txt",'r')
# f[8] = open("OL1/wtc.txt",'r')
# f[9] = open("OL2/wtc.txt",'r')
# f[3] = open("MeredithTLF1MIREX2016/wtc2f20output.txt",'r')
# f[4] = open("MeredithTLPMIREX2016/wtc2f20.tlp",'r')
# f[5] = open("MeredithTLRMIREX2016/wtc2f20.tlr",'r')
# f[6] = open("VM/VM1/output/wtc2f20.txt",'r')
# f[7] = open("VM/VM2/output/wtc2f20.txt",'r')
# f[8] = open("2016IR/bach-mono.txt",'r')

# f={}
# f[0] = open("ref/beethoven-mono.txt",'r')
# f[1] = open("2014ON/bee.txt",'r')
# f[2] = open("2016MP/beethovenOp2No1Mvt3.txt",'r')
# f[8] = open("OL1/sonata01-3.txt",'r')
# f[9] = open("OL2/sonata01-3.txt",'r')
# f[3] = open("MeredithTLF1MIREX2016/sonata01-3.tlf1",'r')
# f[4] = open("MeredithTLPMIREX2016/sonata01-3.tlp",'r')
# f[5] = open("MeredithTLRMIREX2016/sonata01-3.tlr",'r')
# f[6] = open("VM/VM1/output/sonata01-3.txt",'r')
# f[7] = open("VM/VM2/output/sonata01-3.txt",'r')
# # f[8] = open("2016IR/bach-mono.txt",'r')

# f={}
# f[0] = open("ref/mozart-mono.txt",'r')
# f[1] = open("2014ON/mozart.txt",'r')
# f[2] = open("2016MP/mozartK282Mvt2.txt",'r')
# f[8] = open("OL1/sonata04-2.txt",'r')
# f[9] = open("OL2/sonata04-2.txt",'r')
# f[3] = open("MeredithTLF1MIREX2016/sonata04-2.tlf1",'r')
# f[4] = open("MeredithTLPMIREX2016/sonata04-2.tlp",'r')
# f[5] = open("MeredithTLRMIREX2016/sonata04-2.tlr",'r')
# f[6] = open("VM/VM1/output/sonata04-2.txt",'r')
# f[7] = open("VM/VM2/output/sonata04-2.txt",'r')
# # f[8] = open("2016IR/bach-mono.txt",'r')




# f={}
# f[0] = open("ref/gibbons-mono.txt",'r')
# f[1] = open("2014ON/gibbons.txt",'r')
# f[2] = open("2016MP/gibbonsSilverSwan1612.txt",'r')
# f[8] = open("OL1/silverswan.txt",'r')
# f[9] = open("OL2/silverswan.txt",'r')
# f[3] = open("MeredithTLF1MIREX2016/silverswan.tlf1",'r')
# f[4] = open("MeredithTLPMIREX2016/silverswan.tlp",'r')
# f[5] = open("MeredithTLRMIREX2016/silverswan.tlr",'r')
# f[6] = open("VM/VM1/output/silverswan.txt",'r')
# f[7] = open("VM/VM2/output/silverswan.txt",'r')
# # f[8] = open("2016IR/out.txt",'r')

d={}
for i in range(startindex,endindex):
    d[i]=f[i].readlines()

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

# stats calculation
# overlap percentage
def overlap(startflat, endflat, flattened_list):
    overlaptotal=[]
    for i in range(0, len(startflat)):
        overlappatlen=[]
        for pair in flattened_list:
            if startflat[i]>pair[0] and startflat[i]<pair[1] and endflat[i]>=pair[1]:
                overlaplen = pair[1]-startflat[i]
                overlappatlen.append(overlaplen)
            if endflat[i]>pair[0] and endflat[i]<pair[1] and startflat[i]<=pair[0]:
                overlaplen = endflat[i] - pair[0]
                overlappatlen.append(overlaplen)
            if startflat[i]>pair[0] and endflat[i]<pair[1]:
                overlaplen = endflat[i] -startflat[i]
                overlappatlen.append(overlaplen)
            if startflat[i]<pair[0] and endflat[i]>pair[1]:
                overlaplen = endflat[i] - startflat[i]
                overlappatlen.append(overlaplen)
        overlaptotal.append(overlappatlen)
    return overlaptotal

# coverage
def merge_intervals(intervals):
    s = sorted(intervals, key=lambda t: t[0])
    m = 0
    for  t in s:
        if t[0] > s[m][1]:
            m += 1
            s[m] = t
        else:
            s[m] = (s[m][0], t[1])
    return s[:m+1]

startendpat={}
flattened_list={}
startflat={}
endflat={}
totallist=[]
totalstartlist=[]
totalendlist=[]
overlaptotal={}
coverinterval={}
allpitches={}
alltimes={}
for i in range(startindex,endindex):
    startendpat[i]=outputtimes(d[i])
    allpitches[i]=outputtimeandpitch(d[i])[1]
    alltimes[i]=outputtimeandpitch(d[i])[2]
    if startendpat[i]==[]:
        startendpat[i]=[[[0,1]]]
    flattened_list[i]=[y for x in startendpat[i] for y in x]
    startflat[i]=zip(*flattened_list[i])[0]
    endflat[i]=zip(*flattened_list[i])[1]
    totallist=totallist+flattened_list[i]
    totalstartlist=totalstartlist+list(startflat[i])
    totalendlist=totalendlist+list(endflat[i])
    overlaptotal[i]=overlap(startflat[i],endflat[i],flattened_list[i])
    coverinterval[i] = merge_intervals(flattened_list[i])
    print(len(startendpat[i]))



totaltime = max(totalendlist) - min(totalendlist)

overlaptotalper={}
for i in range(startindex,endindex):
    overlaptotalper[i]= [x / totaltime for x in [y for x in overlaptotal[i] for y in x]]
    print("Total overlap length (unit of the piece len):"+str(sum(overlaptotalper[i])))

alloverlap = overlap(totalstartlist,totalendlist,totallist)
alloverlapper = [x / totaltime for x in [y for x in alloverlap for y in x]]
print("all overlap"+str(sum(alloverlapper)))

for i in range(startindex,endindex):
    sum=0
    for interval in coverinterval[i]:
        sum=sum+interval[1] - interval[0]
    print("coverage:"+str(sum/totaltime))

# Ploting
import matplotlib.pyplot as plt
import numpy
height = 0
from operator import itemgetter

c=numpy.random.rand(3,1)
for color in c:
    if color < 0.5:
        color = color + 0.5
fig=plt.figure(figsize=(10,15))
ax=fig.add_subplot(1, 1, 1)



c=numpy.random.rand(3,1)
for color in c:
    if color < 0.5:
        color = color + 0.5
# for patterns in startendpat[1]:
#     # c=numpy.random.rand(3,1)
#     height = height + 1
#     for occur in patterns:
#         plt.plot((occur[0], occur[1]), (height, height), color = c, lw=2, alpha=0.5)
# for n in range(0,len(startendpat[1])):
#     patterns = startendpat[1][n]
#     t=alltimes[1][n]
#     p=allpitches[1][n]


# plt.axhline(y=height+0.5)








# rankinglist=[]
# indexlist=[]
# pickinglist=[]
# for p in range(0,len(startendpat[5])):
#     rankinglist.append(startendpat[5][p][0][0])
#     indexlist.append(p)
# zippedlist=zip(rankinglist,indexlist)
# for pairs in sorted(zippedlist):
#     pickinglist.append(pairs[1])

# print(pickinglist)

# for patterns in startendpat[5]:
#     # c=numpy.random.rand(3,1)
#     height = height + 1
#     for occur in patterns:
#         plt.plot((occur[0], occur[1]), (height, height), color = c, lw=2, alpha=0.5)
# plt.plot((0,0), (0,0), color=c, label="SIATECCompress-TLR")

# c=numpy.random.rand(3,1)
# for color in c:
#     if color < 0.5:
        # color = color + 0.5
# for patterns in startendpat[6]:
#     # c=numpy.random.rand(3,1)
#     height = height + 1
#     for occur in patterns:
#         plt.plot((occur[0], occur[1]), (height, height), color = c, lw=2, alpha=0.5)


c=numpy.random.rand(3,1)
c=(0,0,0)
for color in c:
    if color < 0.5:
        color = color + 0.5
# for n in range(0,len(startendpat[0])):
    # patterns = startendpat[0][n]
    # t=alltimes[0][n]
    # p=allpitches[0][n]
for patterns in startendpat[0]:
    height = height + 1
    # for j in range(0, len(patterns)):
        # occur = patterns[j]
        # tcur = t[j]
        # pcur = p[j]
        # plt.scatter(tcur, [pitch * 0.008 + height for pitch in pcur], color = c, alpha=0.5, s = 3)
        # plt.xlabel('Time')

    for occur in patterns:
        plt.plot((occur[0], occur[1]), (height, height), color = c, lw=2)
plt.plot((0,0), (0,0), color=c, label="Ground Truth")

c=numpy.random.rand(3,1)
for color in c:
    if color < 0.5:
        color = color + 0.5



c=numpy.random.rand(3,1)
c=(0.5,0.5,0.2)
for color in c:
    if color < 0.5:
        color = color + 0.5
for patterns in startendpat[10]:
    # c=numpy.random.rand(3,1)
    height = height + 1
    for occur in patterns:
        plt.plot((occur[0], occur[1]), (height, height), color = c, lw=2, alpha=0.5)
# for n in range(0,len(startendpat[10])):
#     patterns = startendpat[10][n]
#     t=alltimes[10][n]
#     p=allpitches[10][n]

# for patterns in startendpat[0]:
    # height = height + 1
    # for j in range(0, len(patterns)):
    #     occur = patterns[j]
    #     tcur = t[j]
    #     pcur = p[j]
    #     plt.scatter(tcur, [pitch * 0.008 + height for pitch in pcur], color = c, alpha=0.5, s = 3)
    #     plt.xlabel('Time')
plt.plot((0,0), (0,0), color=c, label="SIARCT-CFP(SIACFP)")

c=(0.1,0.1,0.1)
for patterns in startendpat[1]:
    height = height + 1
    # for j in range(0, len(patterns)):
    #     occur = patterns[j]
    #     tcur = t[j]
    #     pcur = p[j]
    #     plt.scatter(tcur, [pitch * 0.008 + height for pitch in pcur], color = c, alpha=0.5, s = 3)
    #     plt.xlabel('Time')

    for occur in patterns:
        plt.plot((occur[0], occur[1]), (height, height), color = c, lw=2, alpha=0.5)
plt.plot((0,0), (0,0), color=c, label="MotivesExtractor(ME)")

c=numpy.random.rand(3,1)
c=(0.5,0.5,0.2)
for color in c:
    if color < 0.5:
        color = color + 0.5
for patterns in startendpat[2]:
    # c=numpy.random.rand(3,1)
    height = height + 1
    for occur in patterns:
        plt.plot((occur[0], occur[1]), (height, height), color = c, lw=2, alpha=0.5)
# plt.axhline(y=height+0.5)
# for n in range(0,len(startendpat[2])):
#     patterns = startendpat[2][n]
#     t=alltimes[2][n]
#     p=allpitches[2][n]

# for patterns in startendpat[0]:
    # height = height + 1
    # for j in range(0, len(patterns)):
    #     occur = patterns[j]
    #     tcur = t[j]
    #     pcur = p[j]
    #     plt.scatter(tcur, [pitch * 0.008 + height for pitch in pcur], color = c, alpha=0.5, s = 3)
    #     plt.xlabel('Time')
plt.plot((0,0), (0,0), color=c, label="SYMCHM(SC)")

c=numpy.random.rand(3,1)
c=(0.1,0.1,0.1)
for color in c:
    if color < 0.5:
        color = color + 0.5
for patterns in startendpat[9]:
    # c=numpy.random.rand(3,1)
    height = height + 1
    for occur in patterns:
        plt.plot((occur[0], occur[1]), (height, height), color = c, lw=2, alpha=0.5)
# for n in range(0,len(startendpat[9])):
#     patterns = startendpat[9][n]
#     t=alltimes[9][n]
#     p=allpitches[9][n]

# for patterns in startendpat[0]:
    # height = height + 1
    # for j in range(0, len(patterns)):
    #     occur = patterns[j]
    #     tcur = t[j]
    #     pcur = p[j]
    #     plt.scatter(tcur, [pitch * 0.008 + height for pitch in pcur], color = c, alpha=0.5, s = 3)
    #     plt.xlabel('Time')
plt.plot((0,0), (0,0), color=c, label="OL2")

c=numpy.random.rand(3,1)
c=(0.5,0.5,0.2)
for color in c:
    if color < 0.5:
        color = color + 0.5
for patterns in startendpat[8]:
    # c=numpy.random.rand(3,1)
    height = height + 1
    for occur in patterns:
        plt.plot((occur[0], occur[1]), (height, height), color = c, lw=2, alpha=0.5)
# for n in range(0,len(startendpat[8])):
#     patterns = startendpat[8][n]
#     t=alltimes[8][n]
#     p=allpitches[8][n]
# for patterns in startendpat[0]:
    # height = height + 1
    # for j in range(0, len(patterns)):
    #     occur = patterns[j]
    #     tcur = t[j]
    #     pcur = p[j]
    #     plt.scatter(tcur, [pitch * 0.008 + height for pitch in pcur], color = c, alpha=0.5, s = 3)
    #     plt.xlabel('Time')
plt.plot((0,0), (0,0), color=c, label="OL1")

c=numpy.random.rand(3,1)
c=(0.1,0.1,0.1)
for color in c:
    if color < 0.5:
        color = color + 0.5
for patterns in startendpat[7]:
    # c=numpy.random.rand(3,1)
    height = height + 1
    for occur in patterns:
        plt.plot((occur[0], occur[1]), (height, height), color = c, lw=2, alpha=0.5)
# for n in range(0,len(startendpat[7])):
#     patterns = startendpat[7][n]
#     t=alltimes[7][n]
#     p=allpitches[7][n]
# for patterns in startendpat[0]:
    # height = height + 1
    # for j in range(0, len(patterns)):
    #     occur = patterns[j]
    #     tcur = t[j]
    #     pcur = p[j]
    #     plt.scatter(tcur, [pitch * 0.008 + height for pitch in pcur], color = c, alpha=0.5, s = 3)
    #     plt.xlabel('Time')
plt.plot((0,0), (0,0), color=c, label="VM2")

c=numpy.random.rand(3,1)
c=(0.5,0.5,0.2)
for color in c:
    if color < 0.5:
        color = color + 0.5
for patterns in startendpat[6]:
    height = height + 1
    for occur in patterns:
        plt.plot((occur[0], occur[1]), (height, height), color = c, lw=2, alpha=0.5)
# for n in range(0,len(startendpat[6])):
#     patterns = startendpat[6][n]
#     t=alltimes[6][n]
#     p=allpitches[6][n]
# for patterns in startendpat[0]:
    # height = height + 1
    # for j in range(0, len(patterns)):
    #     occur = patterns[j]
    #     tcur = t[j]
    #     pcur = p[j]
    #     plt.scatter(tcur, [pitch * 0.008 + height for pitch in pcur], color = c, alpha=0.5, s = 3)
    #     plt.xlabel('Time')
plt.plot((0,0), (0,0), color=c, label="VM1")

c=numpy.random.rand(3,1)
c=(0.1,0.1,0.1)
for color in c:
    if color < 0.5:
        color = color + 0.5
for patterns in startendpat[5]:
    height = height + 1
    for occur in patterns:
        plt.plot((occur[0], occur[1]), (height, height), color = c, lw=2, alpha=0.5)
# for n in range(0,len(startendpat[5])):
#     patterns = startendpat[5][n]
#     t=alltimes[5][n]
#     p=allpitches[5][n]

# # for patterns in startendpat[0]:
#     height = height + 1
#     for j in range(0, len(patterns)):
#         occur = patterns[j]
#         tcur = t[j]
#         pcur = p[j]
#         plt.scatter(tcur, [pitch * 0.008 + height for pitch in pcur], color = c, alpha=0.5, s = 1)
#         plt.xlabel('Time')
plt.plot((0,0), (0,0), color=c, label="SIATECCompress-TLR(SIAR)")   

c=numpy.random.rand(3,1)
c=(0.5,0.5,0.2)
for color in c:
    if color < 0.5:
        color = color + 0.5
for patterns in startendpat[4]:
    # c=numpy.random.rand(3,1)
    height = height + 1
    for occur in patterns:
        plt.plot((occur[0], occur[1]), (height, height), color = c, lw=2, alpha=0.5)
# for n in range(0,len(startendpat[4])):
#     patterns = startendpat[4][n]
#     t=alltimes[4][n]
#     p=allpitches[4][n]

# # for patterns in startendpat[0]:
#     height = height + 1
#     for j in range(0, len(patterns)):
#         occur = patterns[j]
#         tcur = t[j]
#         pcur = p[j]
#         plt.scatter(tcur, [pitch * 0.008 + height for pitch in pcur], color = c, alpha=0.5, s = 3)
#         plt.xlabel('Time')
plt.plot((0,0), (0,0), color=c, label="SIATECCompress-TLP(SIAP)")

c=numpy.random.rand(3,1)
c=(0.1,0.1,0.1)
for color in c:
    if color < 0.5:
        color = color + 0.5

for patterns in startendpat[3]:
    # c=numpy.random.rand(3,1)
    height = height + 1
    for occur in patterns:
        plt.plot((occur[0], occur[1]), (height, height), color = c, lw=2, alpha=0.5)
# plt.axhline(y=height+0.5)
# for n in range(0,len(startendpat[3])):
#     patterns = startendpat[3][n]
#     t=alltimes[3][n]
#     p=allpitches[3][n]

# for patterns in startendpat[0]:
    # height = height + 1
    # for j in range(0, len(patterns)):
    #     occur = patterns[j]
    #     tcur = t[j]
    #     pcur = p[j]
    #     plt.scatter(tcur, [pitch * 0.008 + height for pitch in pcur], color = c, alpha=0.5, s = 3)
    #     plt.xlabel('Time')
plt.plot((0,0), (0,0), color=c, label="SIATECCompress-TLF1(SIAF1)")



# c=numpy.random.rand(3,1)
# for patterns in startendpat[10]:
#     # c=numpy.random.rand(3,1)
#     height = height + 1
#     for occur in patterns:
#         plt.plot((occur[0], occur[1]), (height, height), color = c, lw=2)
# plt.plot((0,0), (0,0), color=c, label="IR")




plt.xlim([0, 600])
plt.ylim([0,153])

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1],loc='best',fontsize=15)
# plt.title('Pattern Visualisation')
plt.xlabel('Time')
plt.ylabel('Patterns and their occurrences')
plt.tight_layout()
plt.tick_params(axis='both', left='off', top='off', right='off', labelleft='off', labeltop='off', labelright='off')
# plt.savefig('chopinwithTom.png')
plt.show()

# print(overlaptotal)
# plt.figure()
# color=numpy.random.rand(3,1)
# print(color)
# color=numpy.array([0.5,0.5,1])
# for x in range(0, len(overlaptotal)):
#     color[2] = color[2] - 0.01
#     print(color)
#     for p in range(0,len(overlaptotal[x])):
#         plt.scatter(x, overlaptotal[x][p],color=color)
# plt.xlabel('Occurrencies ')
# plt.ylabel('Overlapping time')
# plt.show()
