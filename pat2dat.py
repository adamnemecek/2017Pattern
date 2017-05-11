from __future__ import division
import os
import fnmatch

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
for i in range(0,m-1):
    d[i]=f[i].readlines()
    if d[i] != []:
        startendpat[i]=outputtimes(d[i])

print(startendpat)