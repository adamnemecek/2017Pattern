from __future__ import division
startindex=0
endindex=10
# read files to a nice format
profoliolist=[0,1,2,3,4,5,6,7,8,9,10]
weightlist=[0,1,1,1,0.5,0.5,0.3333,0.33333,0.3333,0.5,0.5]
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
# f[10] = open("2016IR/chopin-mono.txt",'r')
f[10] = open("SIARCT-CFP/examples/exampleData/patterns_mazurka24-4.txt")
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
            if pairs != []	:
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

import numpy
dist=numpy.zeros(int(max(totalendlist)))
for time in [x * 1 for x in range(0, int(max(totalendlist)))]:
    for i in range(startindex,endindex):
        for j in range(0,len(startflat[i])):
            if time >= startflat[i][j] and time <=  endflat[i][j]:
                dist[time] = dist[time] + 1 * weightlist[i]

import numpy as np
from math import factorial
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


smoothmax=6
maxthre = 10
reso = 0.1
F13d = numpy.zeros((smoothmax,maxthre))
allcount3d = numpy.zeros((smoothmax,maxthre))
recall3d=numpy.zeros((smoothmax,maxthre))
precision3d=numpy.zeros((smoothmax,maxthre))
for smooth in range(0,smoothmax):
    for i in range(0, smooth):
        dist = savitzky_golay(dist, 3, 1)

    allcount=[]
    recall=[]
    precision=[]
    F1=[]
    for thre in (x * reso for x in range(0, maxthre)):
        deri=[]
        deri2=[]
        corner=[]
        for i in range(0,len(dist)-1):
            deri.append(dist[i+1] - dist[i])

        for i in range(0,len(deri)-1):
            deri2.append(deri[i+1] - deri[i])

        count=0
        cornertime=[]
        cornernumeric=[]
        for i in range(0,len(deri2)-1):
            if (deri2[i+1]>thre and deri2[i] < -thre) or (deri2[i+1]<-thre and deri2[i] >thre):
                cornertime.append(i)
                corner.append(1)
                cornernumeric.append(1)
                count=count+1
            else:
                corner.append(numpy.nan)
                cornernumeric.append(0)
            # deri2.append(deri[i+1] - deri[i]+50)

        # for i in range(0,len(deri2)-1):
        #     if (deri2[i+1]>thre+50 and deri2[i] < -thre+50) or (deri2[i+1]<-thre+50 and deri2[i] >thre+50):
        #         corner.append(1)
        #         count=count+1
        #     else:
        #         corner.append(numpy.nan)

        if sum(cornernumeric) != 0:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(dist)
            plt.plot(numpy.array(deri)*10,c='r',label='First Derivatives')
            plt.plot(numpy.array(deri2)*50,c='g', label='Second Derivatives')
            for i in range(0, len(corner)):
                if corner[i]==1:
                    plt.axvline(i, color='k', linestyle='--')

            GTbd=[]
            height=50
            for patterns in startendpat[0]:
                # c=numpy.random.rand(3,1)
                height = height + 10
                for occur in patterns:
                    plt.plot((occur[0], occur[1]), (height, height), color = 'black', lw=2, alpha=0.5)
                    GTbd.append(occur[0])
                    GTbd.append(occur[1])

            plt.axhline(0)
            plt.plot((0,0), (0,0),color = 'black', label="Ground Truth")
            plt.ylabel('Pattern number & Ground Truth Patterns')
            plt.xlabel('Time')
            plt.title('Extracted boundaries')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig('boundaries/WSupprtsmoothed'+str(smooth)+'-3thre'+str(thre)+str(thre)+str(thre)+str(thre)+'.png')
            allcount.append(count)

            right = 0
            for GT in GTbd:
                for cor in cornertime:
                    if abs(GT - cor) < 1:
                        right = right + 1

            assert right <= len(GTbd)

            rec = right / len(GTbd)
            pre = right / sum(cornernumeric)
            recall.append(rec)
            precision.append(pre)
            
            recall3d[smooth][int(thre*10)] = rec
            precision3d[smooth][int(thre*10)] = pre

            if rec + pre == 0:
                F1.append(0)
                F13d[smooth][int(thre*10)]=0
            else:
                F1.append((2*(pre*rec) / (pre + rec)))
                F13d[smooth][int(thre*10)]=(2*(pre*rec) / (pre + rec))
        else:
            allcount.append(0)
            recall.append(0)
            precision.append(0)
            F1.append(0)

    fig, ax1 = plt.subplots()
    X=numpy.array(range(0,maxthre))*reso
    print(X)
    print(allcount)
    ax1.plot(X, allcount, 'b-')
    ax1.set_xlabel('Threshold (Quarter Length)')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Boundary number', color='purple')
    ax1.tick_params('y', colors='purple')
    
    ax2 = ax1.twinx()
    print(len(X))
    print(len(recall))
    ax2.plot(X, recall, 'r--', label='recall')
    ax2.plot(X, precision, 'g--', label='precision')
    ax2.plot(X, F1, 'b--',label='F1')
    ax2.set_ylabel('Percentage', color='r')
    ax2.tick_params('y', colors='r')
    plt.legend(loc='best')
    plt.title('The effect of the derivatives threshold')
    plt.tight_layout()
    plt.savefig('boundariessmoo'+str(smooth)+'maxthre'+str(maxthre)+'.png')

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X=numpy.array(range(0,maxthre))*reso
# Y=numpy.array(range(0,smoothmax))
# X, Y = np.meshgrid(X, Y)
# Z = numpy.array(precision3d)
# Z2 = numpy.array(F13d)
# Z3 = numpy.array(recall3d)
# print(Z)

# surf = ax.plot_surface(X, Y, Z, cmap='Greens')
# surf2 = ax.plot_surface(X, Y, Z2, cmap='Blues')
# surf3 = ax.plot_surface(X, Y, Z3, cmap='Oranges')

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.set_xlabel('Threshold')
# ax.set_ylabel('Smoothing')
# ax.set_zlabel('Percentage')
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()
