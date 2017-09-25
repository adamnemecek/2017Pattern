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
path =  "C:/Users/admin_local/Desktop/filesboot/nlb/AnnotatedMotifs/discovery/"
path2 = "C:/Users/admin_local/Desktop/filesboot/nlb/InterOpusDiscoveryClassTask/SIATECAlgorithm/false_true_0_3_0.7_5/discovery/"

oneFamilyPath = []
twoFamilyPath = []
for root, dirs, files in os.walk(path):
    for fileiter in files:
        if fnmatch.fnmatch(fileiter, "Daar_ging_een_heer_1*"):
            address= os.path.join(root, fileiter)
            oneFamilyPath.append(address)

for root, dirs, files in os.walk(path2):
    for fileiter in files:
        if fnmatch.fnmatch(fileiter, "Daar_ging_een_heer_1*"):
            address= os.path.join(root, fileiter)
            twoFamilyPath.append(address)

familyData=[]
# Plot discovered patterns in concatenate version
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

familyData2 = []
for path in oneFamilyPath:
    startendpat = outputtimes(open(path,'r').readlines())
    familyData.append(startendpat)


for path in twoFamilyPath:
    startendpat = outputtimes(open(path,'r').readlines())
    familyData2.append(startendpat)

plt.figure()
height=5
starttime=0
index = 0

for Data in familyData:
    for patterns in Data:
        # c=numpy.random.rand(3,1)
        height = height + 10
        for occur in patterns:
            # print(occur)
            plt.plot((occur[0]+starttime, occur[1]+starttime), (height, height), color = 'red', lw=2, alpha=0.5)
    cuetime = Cues[index]
    starttime = cuetime
    index += 1

index = 0
for Data in familyData2:
    for patterns in Data:
        # c=numpy.random.rand(3,1)
        height = height + 10
        for occur in patterns:
            # print(occur)
            plt.plot((occur[0]+starttime, occur[1]+starttime), (height, height), color = 'black', lw=2, alpha=0.5)
    cuetime = Cues[index]
    starttime = cuetime
    index += 1

plt.plot((0,0), (0,0), color='white', label="GT")
for cue in Cues:
    plt.axvline(cue, lw = 1)
plt.ylabel('Pattern Number & Ground Truth Patterns')
plt.xlabel('Time')
# plt.title('The polling curve')
plt.tight_layout()
plt.show()
