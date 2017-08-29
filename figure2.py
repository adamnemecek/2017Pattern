startindex=0
endindex=10
# read files to a nice format
f={}
f[0] = open("ref/chopin-mono.txt",'r')
f[1] = open("2014ON/mazurka24-4-polyoutput.txt",'r')
f[2] = open("2016MP/chopinOp24No4.txt",'r')
f[3] = open("OL1/mazurkaoutput.txt",'r')
f[4] = open("OL2/mazurkaoutput.txt",'r')
f[5] = open("MeredithTLF1MIREX2016/mazurka24-4mono.tlf1",'r')
f[6] = open("MeredithTLPMIREX2016/mazurka24-4.tlf1",'r')
f[7] = open("MeredithTLRMIREX2016/mazurka24-4.tlf1",'r')
f[8] = open("VM/VM1/output/mazurka24-4.txt",'r')
f[9] = open("VM/VM2/output/mazurka24-4input.txt",'r')
f[10] = open("SIARCT-CFP/examples/exampleData/patterns_mazurka24-4.txt")
# f[10] = open("2016IR/chopin-mono.txt",'r')

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


startendpat={}
flattened_list={}
startflat={}
endflat={}
totallist=[]
totalstartlist=[]
totalendlist=[]
overlaptotal={}
coverinterval={}
for i in range(startindex,endindex):
    startendpat[i]=outputtimes(d[i])
    flattened_list[i]=[y for x in startendpat[i] for y in x]
    startflat[i]=zip(*flattened_list[i])[0]
    endflat[i]=zip(*flattened_list[i])[1]
    totallist=totallist+flattened_list[i]
    totalstartlist=totalstartlist+list(startflat[i])
    totalendlist=totalendlist+list(endflat[i])
totaltime = max(totalendlist) - min(totalendlist)

print(startendpat[0])
import numpy
dist=numpy.zeros(int(max(totalendlist)))
for time in [x * 1 for x in range(0, int(max(totalendlist)))]:
    for i in range(startindex,endindex):
        for j in range(0,len(startflat[i])):
            if time >= startflat[i][j] and time <=  endflat[i][j]:
                dist[time] = dist[time] + 1

import matplotlib.pyplot as plt
import numpy
import scipy.stats as ss

plt.figure()
plt.plot(dist)
height=50
for patterns in startendpat[0]:
    # c=numpy.random.rand(3,1)
    height = height + 10
    for occur in patterns:
        print(occur)
        plt.plot((occur[0], occur[1]), (height, height), color = 'red', lw=2, alpha=0.5)
plt.plot((0,0), (0,0), color='white', label="GT")
plt.ylabel('Pattern Number & Ground Truth Patterns')
plt.xlabel('Time')
plt.title('The polling curve')
plt.tight_layout()
plt.show()
# plt.savefig('Supprt.png')
