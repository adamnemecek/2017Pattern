from __future__ import division
profoliolist=[1,2,3,4,5,6,7,8,9,10]
weightlist=[0,1,1,1,0.5,0.5,0.3333,0.33333,0.3333,0.5,0.5]
# profoliolist=[0,1,2,4,5,9]
# profoliolist=[0,2,5,8]
# profoliolist=[0,5,6,7]
# itertools.combinations(iterable, r)

ftotal={}
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
# ftotal['f1'] = f

f={}
f[0] = open("ref/chopin-mono.txt",'r')
f[1] = open("2014ON/mazurka24-4-polyoutput.txt",'r')
f[2] = open("2016MP/chopinOp24No4.txt",'r')
f[3] = open("OL1/chopin1.txt",'r')
f[4] = open("OL2/chopin2.txt",'r')
f[5] = open("MeredithTLF1MIREX2016/mazurka24-4mono.tlf1",'r')
f[6] = open("MeredithTLPMIREX2016/mazurka24-4.tlf1",'r')
f[7] = open("MeredithTLRMIREX2016/mazurka24-4.tlf1",'r')
f[8] = open("VM/VM1/output/mazurka24-4.txt",'r')
f[9] = open("VM/VM2/output/mazurka24-4input.txt",'r')
f[10] = open("SIARCT-CFP/examples/exampleData/patterns_mazurka24-4.txt")
ftotal['f1'] = f

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
# # # f[8] = open("2016IR/bach-mono.txt",'r')
# ftotal['f2'] = f

f={}
f[0] = open("ref/beethoven-mono.txt",'r')
f[1] = open("2014ON/bee.txt",'r')
f[2] = open("2016MP/beethovenOp2No1Mvt3.txt",'r')
f[3] = open("OL1/bee1.txt",'r')
f[4] = open("OL2/beethoven2.txt",'r')
f[5] = open("MeredithTLF1MIREX2016/sonata01-3mono.tlf1",'r')
f[6] = open("MeredithTLPMIREX2016/sonata01-3mono.tlf1",'r')
f[7] = open("MeredithTLRMIREX2016/sonata01-3mono.tlf1",'r')
f[8] = open("VM/VM1/output/sonata01-3.txt",'r')
f[9] = open("VM/VM2/output/sonata01-3.txt",'r')
f[10] = open("SIARCT-CFP/examples/exampleData/patterns_sonata01-3.txt")
# # f[8] = open("2016IR/bach-mono.txt",'r')
ftotal['f2'] = f

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
# ftotal['f3'] = f
# # f[8] = open("2016IR/bach-mono.txt",'r')

f={}
f[0] = open("ref/mozart-mono.txt",'r')
f[1] = open("2014ON/mozart.txt",'r')
f[2] = open("2016MP/mozartK282Mvt2.txt",'r')
f[3] = open("OL1/mozart1.txt",'r')
f[4] = open("OL2/mozart2.txt",'r')
f[5] = open("MeredithTLF1MIREX2016/sonata04-2mono.tlf1",'r')
f[6] = open("MeredithTLPMIREX2016/sonata04-2mono.tlf1",'r')
f[7] = open("MeredithTLRMIREX2016/sonata04-2mono.tlf1",'r')
f[8] = open("VM/VM1/output/sonata04-2.txt",'r')
f[9] = open("VM/VM2/output/sonata04-2.txt",'r')
f[10] = open("SIARCT-CFP/examples/exampleData/patterns_sonata04-2.txt")
ftotal['f3'] = f

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
# # f[8] = open("2016IR/bach-mono.txt",'r')
# ftotal['f4'] = f
#
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
# ftotal['f5'] = f
# # # f[8] = open("2016IR/out.txt",'r')

precision3 = []
recall3 = []
F13=[]
yay=0
allpropF1=[]
allmaxF1=[]
allproppre=[]
allproprec=[]
allmaxprecision=[]
allmaxrecall=[]

gp1=[]
gp2=[]
gr1=[]
gr2=[]
gf1=[]
gf2=[]

import matplotlib.pyplot as plt
plt.figure()
m= 0
for f in ftotal.values():
    if m!=0:

        plt.scatter(range(0,len(profoliolist)),Z, c='green')
        plt.scatter(range(0,len(profoliolist)),Z2, c='blue')
        plt.scatter(range(0,len(profoliolist)),Z3, c='red')

        plt.axhline(maxprecision, c='darkgreen',lw=3,alpha=0.7)
        plt.axhline(maxrecall, c='red',lw=3,alpha=0.7)
        plt.axhline(maxF1,c='darkblue',lw=3,alpha=0.7)
    d={}
    d[0]=f[0].readlines()
    for i in profoliolist:
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
    startendpat[0]=outputtimes(d[0])
    for i in profoliolist:
        startendpat[i]=outputtimes(d[i])
        flattened_list[i]=[y for x in startendpat[i] for y in x]
        startflat[i]=zip(*flattened_list[i])[0]
        endflat[i]=zip(*flattened_list[i])[1]
        totallist=totallist+flattened_list[i]
        totalstartlist=totalstartlist+list(startflat[i])
        totalendlist=totalendlist+list(endflat[i])
    totaltime = max(totalendlist) - min(totalendlist)

    import numpy

    allcount=[]
    recall=[]
    precision=[]
    F1=[]

    for i in profoliolist:
        GTbd=[]
        for patterns in startendpat[0]:
            for occur in patterns:
                GTbd.append(occur[0])
                GTbd.append(occur[1])
        timeset=[]
        dist=numpy.zeros(int(max(totalendlist)))
        distbin=numpy.zeros(int(max(totalendlist)))

        for time in [x * 1 for x in range(0, int(max(totalendlist)))]:
            for j in range(0,len(startflat[i])):
                if time >= startflat[i][j] and time <=  endflat[i][j]:
                    dist[time] = 1
                    distbin[time] = 1


        for j in range(0,len(startflat[i])):
            timeset.append(startflat[i][j])
            timeset.append(endflat[i][j])

        # if i==0:
        #     print(timeset)
        #     print(GTbd)

        import operator
        gap={}
        right = 0
        for cor in timeset:
            for y in range(0,len(GTbd)):
                gap[y]=(abs(cor - GTbd[y]))

            sortedgap = sorted(gap.items(),key=operator.itemgetter(1))

            smallest = GTbd[sortedgap[0][0]]
            if abs(smallest - cor) == 0 :
                right = right + 1

        if i==0:
            print(right)

        assert right <= len(timeset)
        pre = right / len(timeset)

        # assert right <= sum(distbin)
        # pre = right / sum(distbin)

        gap={}
        right = 0
        for GT in GTbd:
            for y in range(0,len(timeset)):
                gap[y]=(abs(timeset[y] - GT))

            sortedgap = sorted(gap.items(),key=operator.itemgetter(1))
            smallest = timeset[sortedgap[0][0]]
            if abs(smallest - GT) < 1:
                right = right + 1

        assert right <= len(GTbd)
        rec = right / len(GTbd)

        if i==0:
            print(len(GTbd))

        recall.append(rec)
        precision.append(pre)


        if rec + pre == 0:
            F1.append(0)
        else:
            F1.append((2*(pre*rec) / (pre + rec)))

    Z = numpy.array(precision)
    # print("check"+str(Z))
    Z2 = numpy.array(F1)
    Z3 = numpy.array(recall)

    F13.append(Z2)
    precision3.append(Z)
    recall3.append(Z3)
    m=m+1

    import numpy
    dist=numpy.zeros(int(max(totalendlist)))
    for time in [x * 1 for x in range(0, int(max(totalendlist)))]:
        for i in profoliolist[1:]:
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


    smoothmax= 30
    maxthre = 100
    reso = 0.01
    F13d = numpy.zeros((smoothmax,maxthre))
    allcount3d = numpy.zeros((smoothmax,maxthre))
    recall3d=numpy.zeros((smoothmax,maxthre))
    precision3d=numpy.zeros((smoothmax,maxthre))
    for smooth in range(0,smoothmax):
        for i in range(0, smooth):
            dist = savitzky_golay(dist, 5, 1)

        allcount=[]
        recall=[]
        precision=[]
        F1=[]
        for thre in (x * reso for x in range(0, maxthre)):
            thre2 = thre/150
            deri=[]
            deri2=[]
            deri2normal=[]
            corner=[]
            for i in range(0,len(dist)-1):
                deri.append(dist[i+1] - dist[i])

            for i in range(0,len(deri)-1):
                deri2.append((deri[i+1] - deri[i])*5-50)
                deri2normal.append(deri[i+1] - deri[i])

            count=0
            cornertime=[]
            cornernumeric=[]
            for i in range(0,len(deri2)-1):
                if (deri[i+1]>thre and deri[i] < -thre) or (deri[i+1]<-thre and deri[i] >thre) or \
                    (deri2normal[i+1]>thre2 and deri2normal[i] < -thre2) or (deri2normal[i+1]<-thre2 and deri2normal[i] >thre2):
                    cornertime.append(i)
                    corner.append(1)
                    cornernumeric.append(1)
                    count=count+1
                else:
                    corner.append(numpy.nan)
                    cornernumeric.append(0)

            if sum(cornernumeric) != 0:
                import matplotlib.pyplot as plt
                GTbd=[]
                height=30
                for patterns in startendpat[0]:
                    for occur in patterns:
                        GTbd.append(occur[0])
                        GTbd.append(occur[1])

                import operator
                gap={}
                right = 0
                for cor in cornertime:
                    for y in range(0,len(GTbd)):
                        gap[y]=(abs(cor - GTbd[y]))

                    sortedgap = sorted(gap.items(),key=operator.itemgetter(1))

                    smallest = GTbd[sortedgap[0][0]]
                    if abs(smallest - cor) == 0:
                        right = right + 1


                assert right <= len(cornertime)
                pre = right /  len(cornertime)

                gap={}
                right = 0
                for GT in GTbd:
                    for y in range(0,len(cornertime)):
                        gap[y]=(abs(cornertime[y] - GT))

                    assert max(gap.keys()) <= len(cornertime)
                    assert len(gap) == len(cornertime)
                    sortedgap = sorted(gap.items(),key=operator.itemgetter(1))
                    smallest = cornertime[sortedgap[0][0]]
                    if abs(smallest - GT) < 1:
                        right = right + 1

                assert right <= len(GTbd)
                rec = right / len(GTbd)

                recall.append(rec)
                precision.append(pre)

                recall3d[smooth][int(thre*100)] = rec
                precision3d[smooth][int(thre*100)] = pre

                if rec + pre == 0:
                    F1.append(0)
                    F13d[smooth][int(thre*100)]=0
                else:
                    F1.append((2*(pre*rec) / (pre + rec)))
                    F13d[smooth][int(thre*100)]=(2*(pre*rec) / (pre + rec))
            else:
                allcount.append(0)
                recall.append(0)
                precision.append(0)
                F1.append(0)

    Zfuse = numpy.array(precision3d)
    Z2fuse = numpy.array(F13d)
    Z3fuse = numpy.array(recall3d)
    for i,j in enumerate(precision3d):
        for k,l in enumerate(j):
            if l>0.5:
                print i,k

    Z2fuseprop=Z2fuse[2][6]
    prefuseprop=Zfuse[2][6]  
    recfuseprop = Z3fuse[2][6]
    maxprecision=numpy.amax(Zfuse)
    maxrecall=numpy.amax(Z3fuse)
    maxF1=numpy.amax(Z2fuse)

    allmaxprecision.append(maxprecision)
    allmaxrecall.append(maxrecall)
    allmaxF1.append(maxF1)
    allpropF1.append(Z2fuseprop)
    allproppre.append(prefuseprop)
    allproprec.append(recfuseprop)

    precorrec = 0
    precorf1 = 0
    for i,j in enumerate(precision3d):
        for k,l in enumerate(j):
            # if l>0.5:
            #     # print i,k
            #     gp1.append(i)
            #     gp2.append(k)
            if l == maxprecision:
                if Z3fuse[i][k] > precorrec:
                    precorrec=Z3fuse[i][k]
                    precorf1=Z2fuse[i][k]
                    gp1.append(i)
                    gp2.append(k)

    reccorpre = 0
    reccorf1 = 0
    for i,j in enumerate(recall3d):
        for k,l in enumerate(j):
            # if l>0.5:
            #     # print i,k
            #     gr1.append(i)
            #     gr2.append(k)
            if l == maxrecall:
                if Zfuse[i][k] > reccorpre:
                    reccorpre=Zfuse[i][k]
                    reccorf1=Z2fuse[i][k]
                    gr1.append(i)
                    gr2.append(k)

    f1corpre = 0
    f1corfrec = 0
    for i,j in enumerate(F13d):
        for k,l in enumerate(j):
            # if l>0.2:
            #     # print i,k
            #     gf1.append(i)
            #     gf2.append(k)
            if l == maxF1:
                f1corpre=Zfuse[i][k]
                f1correc=Z3fuse[i][k]
                gf1.append(i)
                gf2.append(k)

    # for F1s in Z2:
    #     if F1s <= maxF1:
    #         yay=yay+1

    # print('how many precisions: '+str(len(Z)))
    for pres in Z:
        if pres <= maxprecision:
            yay=yay+1


    # print(Z2fuse)
    # print(maxprecision)
    # print(maxrecall)
    # print(maxF1)
print(gf1)
print(gf2)

plt.axhline(maxprecision, c='darkgreen',lw=3,alpha=0.7,label='voting max precision')
plt.axhline(maxrecall, c='red',lw=3,alpha=0.7,label='voting max recall')
plt.axhline(maxF1,c='darkblue',lw=3,alpha=0.7, label='voting max F1')

plt.scatter(range(0,len(profoliolist)),Z,c='green',label='precision')
plt.scatter(range(0,len(profoliolist)),Z2, c='blue',label='F1')
plt.scatter(range(0,len(profoliolist)),Z3, c='red',label='recall')
plt.legend(loc='best')
x = [0,1,2,3,4,5,6,7,8,9]
labels=['MotivesExtractor', 'SYMCHM','OL1','OL2','SIATECCTLF1','SIATECCTLP','SIATECCTLR','VM1','VM2','SIARCT-CFP']
plt.xticks(x, labels)

plt.xlabel('Algorithms')
plt.ylabel('Percentage')
plt.tight_layout()
# plt.show()

variance =[]
for tp in zip(*precision3):
    norm = [float(i)/max(tp) for i in tp]
    variance.append(numpy.var(norm))
    # variance.append(numpy.var(tp))
    # print('var: '+str(numpy.var(norm)))

variancerec =[]
for tp in zip(*recall3):
    norm = [float(i)/max(tp) for i in tp]
    variancerec.append(numpy.var(norm))
    # variancerec.append(numpy.var(tp))
    # print('var: '+str(numpy.var(norm)))

variancef1 =[]
for tp in zip(*F13):
    norm = [float(i)/max(tp) for i in tp]
    variancef1.append(numpy.var(norm))
    # print('var: '+str(numpy.var(norm)))
    # variancef1.append(numpy.var(tp))


norm = [float(i)/max(allmaxF1) for i in allmaxF1]
print("Max value variance = "+str(numpy.var(norm)))

norm = [float(i)/max(allpropF1) for i in allpropF1]
print("Fixed para variance = "+str(numpy.var(norm)))

print("Better than percentage = "+str(yay/30))

# print(Z)
# print(Z2)
# print(Z3)
# print(maxprecision)
# print(maxrecall)
# print(maxF1)
# print(F13)

avgpre=[]
for index in range(0,len(zip(*precision3))):
    avgpre.append(numpy.mean(zip(*precision3)[index]))

avgf1=[]
for index in range(0,len(zip(*F13))):
    avgf1.append(numpy.mean(zip(*F13)[index]))

avgrec=[]
for index in range(0,len(zip(*recall3))):
    avgrec.append(numpy.mean(zip(*recall3)[index]))




textfile=open("output4.txt","w")
for index in range(0,len(Z)+2):
    textfile.write('\hline\n')
    if index==0:
        textfile.write("ME & ("+ str(avgpre[index])[:5]+", "+ str(variance[index])[:5] +") & ("+str(avgrec[index])[:5]+", "+str(variancerec)[1:6]+") & ("+str(avgf1[index])[:5]+", "+str(variancef1)[1:6]+")\\\\"+"\n")
    if index==1:
        # textfile.write("SYMCHM & "+ str(avgpre[index])[:5]+" & "+ str(variance[index])[:5] +"\\\\"+"\n")
        textfile.write("SC & ("+ str(avgpre[index])[:5]+", "+ str(variance[index])[:5] +") & ("+str(avgrec[index])[:5]+", "+str(variancerec[index])[:5]+") & ("+str(avgf1[index])[:5]+", "+str(variancef1[index])[1:6]+")\\\\"+"\n")
    if index==2:
        # textfile.write("OL1 & "+ str(avgpre[index])[:5]+" & "+ str(variance[index])[:5] +"\\\\"+"\n")
        textfile.write("OL1 & ("+ str(avgpre[index])[:5]+", "+ str(variance[index])[:5] +") & ("+str(avgrec[index])[:5]+", "+str(variancerec[index])[:5]+") & ("+str(avgf1[index])[:5]+", "+str(variancef1[index])[:5]+")\\\\"+"\n")
    if index==3:
        # textfile.write("OL2 & "+ str(avgpre[index])[:5]+" & "+ str(variance[index])[:5] +"\\\\"+"\n")
        textfile.write("OL2 & ("+ str(avgpre[index])[:5]+", "+ str(variance[index])[:5] +") & ("+str(avgrec[index])[:5]+", "+str(variancerec[index])[:5]+") & ("+str(avgf1[index])[:5]+", "+str(variancef1[index])[:5]+")\\\\"+"\n")
    if index==4:
        # textfile.write("SIAF1 & "+ str(avgpre[index])[:5]+" & "+ str(variance[index])[:5] +"\\\\"+"\n")
        textfile.write("SIAF1 & ("+ str(avgpre[index])[:5]+", "+ str(variance[index])[:5] +") & ("+str(avgrec[index])[:5]+", "+str(variancerec[index])[:5]+") & ("+str(avgf1[index])[:5]+", "+str(variancef1[index])[:5]+")\\\\"+"\n")
    if index==5:
        # textfile.write("SIAR & "+ str(avgpre[index])[:5]+" & "+ str(variance[index])[:5] +"\\\\"+"\n")
        textfile.write("SIAR & ("+ str(avgpre[index])[:5]+", "+ str(variance[index])[:5] +") & ("+str(avgrec[index])[:5]+", "+str(variancerec[index])[:5]+") & ("+str(avgf1[index])[:5]+", "+str(variancef1[index])[:5]+")\\\\"+"\n")
    if index==6:
        # textfile.write("SIAP & "+ str(avgpre[index])[:5]+" & "+ str(variance[index])[:5] +"\\\\"+"\n")
        textfile.write("SIAP & ("+ str(avgpre[index])[:5]+", "+ str(variance[index])[:5] +") & ("+str(avgrec[index])[:5]+", "+str(variancerec[index])[:5]+") & ("+str(avgf1[index])[:5]+", "+str(variancef1[index])[:5]+")\\\\"+"\n")
    if index==7:
        # textfile.write("VM1 & "+ str(avgpre[index])[:5]+" & "+ str(variance[index])[:5] +"\\\\"+"\n")
        textfile.write("VM1 & ("+ str(avgpre[index])[:5]+", "+ str(variance[index])[:5] +") & ("+str(avgrec[index])[:5]+", "+str(variancerec[index])[:5]+") & ("+str(avgf1[index])[:5]+", "+str(variancef1[index])[:5]+")\\\\"+"\n")
    if index==8:
        # textfile.write("VM2 & "+ str(avgpre[index])[:5]+" & "+ str(variance[index])[:5] +"\\\\"+"\n")
        textfile.write("VM2 & ("+ str(avgpre[index])[:5]+", "+ str(variance[index])[:5] +") & ("+str(avgrec[index])[:5]+", "+str(variancerec[index])[:5]+") & ("+str(avgf1[index])[:5]+", "+str(variancef1[index])[:5]+")\\\\"+"\n")
    if index==9:
        # textfile.write("SIARCT-CFP & "+ str(avgpre[index])[:5]+" & "+ str(variance[index])[:5] +"\\\\"+"\n")
        textfile.write("SIACFP & ("+ str(avgpre[index])[:5]+", "+ str(variance[index])[:5] +") & ("+str(avgrec[index])[:5]+", "+str(variancerec[index])[:5]+") & ("+str(avgf1[index])[:5]+", "+str(variancef1[index])[:5]+")\\\\"+"\n")

    if index==10:
        norm = [float(i)/max(allproppre) for i in allproppre]
        normf1 = [float(i)/max(allpropF1) for i in allpropF1]
        normrec = [float(i)/max(allproprec) for i in allproprec]


        normm = [float(i)/max(allmaxprecision) for i in allmaxprecision]
        normf1m = [float(i)/max(allmaxF1) for i in allmaxF1]
        normrecm = [float(i)/max(allmaxrecall) for i in allmaxrecall]

        # textfile.write("PPA & "+ str(numpy.mean(allproppre))[:5]+" & "+ str(numpy.var(norm))[:5] +"\\\\"+"\n")
        # textfile.write("PPA & ("+ str(numpy.mean(allproppre))[:5]+", "+ str(numpy.var(norm))[:5] +") & ("+str(numpy.mean(allproprec))[:5]+", "+str(numpy.var(normrec))[:5]+") & ("+str(numpy.mean(allpropF1))[:5]+", "+str(numpy.var(normf1))[:5]+")\\\\"+"\n")
        # textfile.write("PPA & ("+ str(maxprecision)[:5]+", "+ str(numpy.var(norm))[:5] +") & ("+str(maxrecall)[:5]+", "+str(numpy.var(normrec))[:5]+") & ("+str(maxF1)[:5]+", "+str(numpy.var(normf1))[:5]+")\\\\"+"\n")
        textfile.write("PPA mp & ("+ str(numpy.mean(allmaxprecision))[:5]+", "+ str(numpy.var(normm))[:5] +") & ("+str(numpy.mean(allmaxrecall))[:5]+", "+str(numpy.var(normrecm))[:5]+") & ("+str(numpy.mean(allmaxF1))[:5]+", "+str(numpy.var(normf1m))[:5]+")\\\\"+"\n")
        # textfile.write("PPA dp & ("+ str(numpy.mean(allmaxprecision))[:5]+", "+ str(numpy.var(allmaxprecision))[:5] +") & ("+str(numpy.mean(allmaxrecall))[:5]+", "+str(numpy.var(allmaxrecall))[:5]+") & ("+str(numpy.mean(allmaxF1))[:5]+", "+str(numpy.var(allmaxF1))[:5]+")\\\\"+"\n")
        # textfile.write("PPA & ("+ str(numpy.mean(allproppre))[:5]+", "+ str(numpy.var(allproppre))[:5] +") & ("+str(numpy.mean(allproprec))[:5]+", "+str(numpy.var(allproprec))[:5]+") & ("+str(numpy.mean(allpropF1))[:5]+", "+str(numpy.var(allpropF1))[:5]+")\\\\"+"\n")

    if index==11:
        textfile.write("PPA fp& ("+ str(numpy.mean(allproppre))[:5]+", "+ str(numpy.var(norm))[:5] +") & ("+str(numpy.mean(allproprec))[:5]+", "+str(numpy.var(normrec))[:5]+") & ("+str(numpy.mean(allpropF1))[:5]+", "+str(numpy.var(normf1))[:5]+")\\\\"+"\n")
     