from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import medfilt
import math
import random
from pyemd import emd

from scipy.spatial.distance import pdist, wminkowski, squareform
from scipy.spatial.distance import pdist
from scipy.spatial import distance_matrix
from sklearn.metrics.pairwise import pairwise_distances, manhattan_distances, euclidean_distances
from scipy.stats import pearsonr

def diagfill(k=3, shape=(10,10)):
    em = np.zeros(shape[0]*shape[0]).reshape((shape[0],shape[0]))
    diag = zip(np.arange(em.shape[0]),np.arange(em.shape[0]))
    for (i,j) in diag[0:em.shape[0]-(k-1)]:
        em[:,i:i+k][i:i+k] = np.ones((k,k))
    return(em)

def paddedcorrelation(mx, kernelsize=8):
    # create a checkerboard kernel
    print(kernelsize/2)
    kernel = np.kron(np.array([1, -1, -1, 1]).reshape((2,2)), np.ones((1,1)))
    # create a partial self-similarity matrix along the diagonal kernel size
    sm = np.zeros((mx.shape[0], mx.shape[0]))
    dix = np.where(diagfill(kernel.shape[0], sm.shape) == 1)
    for (i,j) in zip(dix[0],dix[1]):
        # print(np.sqrt(np.sum((mx[i]-mx[j])**2)))
        sm[i,j] = np.sqrt(np.sum((mx[i]-mx[j])**2)) 
    # pad the edges to the size of the kernel
    smpad = np.zeros((mx.shape[0]+kernel.shape[0],mx.shape[0]+kernel.shape[0]))
    smpad[:,(int(kernel.shape[0]/2)):-(int(kernel.shape[0]/2))][(int(kernel.shape[0]/2)):-(int(kernel.shape[0]/2))] = sm
    # calculate correlation with pointwise multiplication
    corr = np.array([np.multiply(smpad[:,i:i+kernel.shape[0]][i:i+kernel.shape[0]],kernel).sum() for i in np.arange(0,smpad.shape[0]-kernel.shape[0])])
    paddedcorr = np.concatenate((np.zeros(int(kernel.shape[0]/2)),corr,np.zeros(int(kernel.shape[0]/2))))
    corrn = (corr + np.abs(corr.min())) / (corr + np.abs(corr.min())).max()
    return(corrn)

def mean( hist ):
        mean = 0.0;
        for i in hist:
            mean += i;
        mean/= len(hist);
        return mean;

def bhatta ( hist1,  hist2):

    # calculate mean of hist1
    h1_ = mean(hist1);

    # calculate mean of hist2
    h2_ = mean(hist2);
    # if h1_ == 0:
    #     print(hist1)
    # if h2_ ==0:
    #     print(hist2)

    # calculate score
    score = 0;
    for i in range(len(hist1)):
        score += math.sqrt( hist1[i] * hist2[i] );
    # print h1_,h2_,score;
    # try:
    #     score = -np.log(score)
    #     # score = math.sqrt( 1 - ( 1 / math.sqrt(h1_*h2_*len(hist1)*len(hist2)) ) * score );
    # except ValueError:
    #     score = 1
    return score;

scores=[]
scores1=[]
scoresran=[]
scores2=[]
scores3=[]
scores4=[]
scores12=[]
scoressum=[]
scores1_2=[]
emdscoores=[]
emdscores1=[]
emdscoresran=[]
emdscores2=[]
emdscores3=[]
emdscores4=[]
emdscores12=[]
emdscoressum=[]
emdscores1_2=[]
scoresb={}
pprepsum=[]
pprepGT=[]
pprepnovel=[]
pprepalg={}
for algindex in range(0,7):
    pprepalg[algindex] = []

# for i in range(0, 3):
for i in range(0, 360):
    print(i) 

    algs={}
    # algs[0]=pd.read_csv('C:/Users/admin_local/Dropbox/fusion/data/MIREXinput/output' + str(i) + '.txt', header=None, sep=' ')
    algs[1]=pd.read_csv('/Users/irisren/Dropbox/fusion/data/liederenbankinput/output' + str(i) + '.txt', header=None, sep=' ').as_matrix()
    for algindex in range(1,5):
        # algs[algindex] = np.array(pd.read_csv('C:/Users/admin_local/Dropbox/fusion/data/MIREXinput/{0}output'.format(algindex) + str(i) + '.txt', header=None, sep=' '))
        # algs[algindex+1] = pd.read_csv('/Users/irisren/Dropbox/fusion/data/MIREXinput/{0}output'.format(algindex) + str(i) + '.txt', header=None, sep=' ').as_matrix()
        algs[algindex+1] = pd.read_csv('/Users/irisren/Dropbox/fusion/data/liederenbankinput/{0}output'.format(algindex) + str(i) + '.txt', header=None, sep=' ').as_matrix()

    # algs[6] = pd.read_csv('/Users/irisren/Dropbox/fusion/data/liederenbankinput/12output' + str(i) + '.txt', header=None, sep=' ').as_matrix()
    # pdout = pd.read_csv('C:/Users/admin_local/Dropbox/fusion/data/MIREXinput/output' + str(i) + '.txt', header=None, sep=' ')
    # pdout = pd.read_csv('/Users/irisren/Dropbox/fusion/data/MIREXinput/output' + str(i) + '.txt', header=None, sep=' ')
    pdout = pd.read_csv('/Users/irisren/Dropbox/fusion/data/liederenbankinput/output' + str(i) + '.txt', header=None, sep=' ')
    # pdout1 = pd.read_csv('C:/Users/admin_local/Dropbox/fusion/data/MIREXinput/1output' + str(i) + '.txt', header=None, sep=' ')
    # pdout2 = pd.read_csv('C:/Users/admin_local/Dropbox/fusion/data/MIREXinput/2output' + str(i) + '.txt', header=None, sep=' ')
    # pdout3 = pd.read_csv('C:/Users/admin_local/Dropbox/fusion/data/MIREXinput/3output' + str(i) + '.txt', header=None, sep=' ')
    # pdout4 = pd.read_csv('C:/Users/admin_local/Dropbox/fusion/data/MIREXinput/4output' + str(i) + '.txt', header=None, sep=' ')
    # pdout12 = pd.read_csv('C:/Users/admin_local/Dropbox/fusion/data/MIREXinput/12output' + str(i) + '.txt', header=None, sep=' ')

    mxnp = np.array(pdout)
    # mxnp1 = np.array(pdout1)
    # mxnp2 = np.array(pdout2)
    # mxnp3 = np.array(pdout3)
    # mxnp4 = np.array(pdout4)
    # mxnp12 = np.array(pdout12)

    # pdout = pd.read_csv('C:/Users/admin_local/Dropbox/fusion/data/MIREXinput/output' + str(i) + '.txt', header=None, sep=' ')
    # mxnp = np.array(pdout)

    try:
        # gt = pd.read_csv('C:/Users/admin_local/Dropbox/fusion/data/MIREXinput/GT' + str(i) + '.txt', header=None, sep=' ')
        # gt = pd.read_csv('/Users/irisren/Dropbox/fusion/data/MIREXinput/GT' + str(i) + '.txt', header=None, sep=' ')
        gt = pd.read_csv('/Users/irisren/Dropbox/fusion/data/liederenbankinput/GT' + str(i) + '.txt', header=None, sep=' ')
        # gt = pd.read_csv('C:/Users/admin_local/Dropbox/fusion/data/MIREXinput/GT' + str(i) + '.txt', header=None, sep=' ')
    except:
        print("problem")
        continue

    gtmx = np.array(gt)

    # print(len(mxnp[0]))
    np.set_printoptions(threshold=np.nan)

    distancelist=[]
    for l in range(0,len(mxnp[0])):
        distancelist.append([l]+range(1,len(mxnp[0])))

    distancematrix = euclidean_distances(distancelist)
    # if i==1:
    #     print(distancematrix)
    # print(distancematrix.shape)
    # mxall = np.vstack((mxnp,gtmx+5))
    # corrs = np.array([paddedcorrelation(mxnp.T, kernelsize=i) for i in np.arange(2,10,1)])

    corrs = {}
    for argindex in range(1,6):
        corrs[argindex] = np.array(paddedcorrelation(algs[argindex].T,kernelsize=2))

    if math.isnan(corrs[2][0]):
        continue
        print("nan encountered")

    if math.isnan(corrs[3][0]):
        continue
        print("nan encountered")

    if math.isnan(corrs[4][0]):
        continue
        print("nan encountered")

    if math.isnan(corrs[5][0]):
        continue
        print("nan encountered")   

    for argindex in range(1,6):
        corrs[argindex] = corrs[argindex]/sum(corrs[argindex])
        for item in corrs[argindex]:
            pprepalg[argindex].append(item)

    corrs[6] = np.array(np.sum(mxnp, axis=0))
    corrs[6] = corrs[6]/sum(corrs[6])

    for item in corrs[6]:
        pprepalg[6].append(item)
    corrsold = np.array(paddedcorrelation(mxnp.T, kernelsize=2))
    # corrs1 = np.array(paddedcorrelation(mxnp1.T, kernelsize=2))
    # corrs2 =np.array(paddedcorrelation(mxnp2.T, kernelsize=2))
    # corrs3 =np.array(paddedcorrelation(mxnp3.T, kernelsize=2))
    # corrs4 =np.array(paddedcorrelation(mxnp4.T, kernelsize=2))
    # corrs12 =np.array(paddedcorrelation(mxnp12.T, kernelsize=2))

    
    sumall = np.array(np.sum(mxnp, axis=0))
    sumall = sumall/sum(sumall)
    
    gtdist = np.sum(gtmx, axis=0)/np.sum(np.sum(gtmx, axis=0))

    corrs[0] = gtdist
    for item in corrs[0]:
        pprepalg[0].append(item)
    corrsold = corrsold/sum(corrsold)
    # corrs1 = corrs1/sum(corrs1)
    # corrs2 = corrs2/sum(corrs2)
    # corrs3 = corrs3/sum(corrs3)
    # corrs4 = corrs4/sum(corrs4)
    # corrs12 = corrs12/sum(corrs12)

    scoresb[i] = np.empty(shape=(7,7))
    for m in range(0,7):
        for n in range(0,7):
            scoresb[i][m,n] = bhatta(corrs[m], corrs[n])

temp=np.zeros(shape=(7,7))
for matrix in scoresb.values():
    temp = np.add(temp, matrix)
temp = temp / 360
np.set_printoptions(threshold=np.nan)
print(temp)

# fig = plt.figure()
# ax1 = plt.subplot(1, 2, 1)
# fig = plt.figure()

pearsonindex=np.zeros(shape=(7,7))
pearsonindexp=np.zeros(shape=(7,7))
for m in range(0,7):
    for n in range(0,7):
        pearsonindex[m,n]=pearsonr(pprepalg[m],pprepalg[n])[0]
        pearsonindexp[m,n]=pearsonr(pprepalg[m],pprepalg[n])[1]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
# plt.subplot(121)
plt.sca(axes[0])
labels = ["GT", "Novelty", "PatMinr", "ME", "COSIATEC", "MGDP", "PC"]
plt.xticks(np.arange(0, 7, 1),rotation='vertical', fontsize=14)
plt.gca().set_xticklabels(labels)
labels = ["GT", "Novelty", "PatMinr", "ME", "COSIATEC", "MGDP", "PC"]
plt.yticks(np.arange(0, 7, 1), fontsize=14)
plt.gca().set_yticklabels(labels)
plt.tight_layout()
plt.imshow(pearsonindex, vmin=0, vmax=1, aspect='auto') 


plt.sca(axes[1])

plt.imshow(temp, vmin=0, vmax=1, aspect='auto')
labels = ["GT", "Novelty", "PatMinr", "ME", "COSIATEC", "MGDP", "PC"]
plt.xticks(np.arange(0, 7, 1),rotation='vertical', fontsize=14)
plt.gca().set_xticklabels(labels)
plt.yticks([])

# ax2 = plt.subplot(1, 2, 2)
# im = ax2.imshow(temp)
# cbar_ax = fig.add_axes([0.55, 0.15, 0.05, 0.7])
# fig.subplots_adjust(bottom=0.8)
# plt.colorbar(im, ticks=[-1, 0, 1], orientation='horizontal')
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar( cax=cax)
plt.tight_layout()
plt.show()

# print(pearsonindex)
# print(pearsonindexp)

#     scores.append(bhatta(corrs, gtdist)) 
#     random = np.random.randint(1000,size=len(gtdist))
#     norrandom=random/sum(random)
   
    # plt.figure(figsize=(10,5))
    # plt.yticks([])
    # # plt.plot(norrandom, label='Ran')
    # # plt.plot(corrs1, label='A1')
    # plt.plot(corrsold, label = 'N - TINA output',lw=1,color='orange',alpha=0.8)
    # plt.plot(gtdist,label='GT - Human annotation',lw=2,color='g',alpha=0.8)
    # plt.plot(sumall,label='PC - TINA output',lw=2,color='r',alpha=0.8)
    # plt.xlabel("Time")
    # plt.ylabel("P")
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('distMIREX'+str(i))

    # scoresran.append(bhatta(norrandom,gtdist))
    # scores1.append(bhatta(corrs1,gtdist))
    # scores2.append(bhatta(corrs2,gtdist))
    # scores3.append(bhatta(corrs3,gtdist))
    # scores4.append(bhatta(corrs4,gtdist))
#     # scores12.append(bhatta(corrs12,gtdist))
#     scoressum.append(bhatta(sumall,gtdist))
#     # scores1_2.append(bhatta(corrs1,corrs2))
#     emdscoores.append(emd(corrs, gtdist, distancematrix))
#     # emdscores1.append(emd(corrs1, gtdist,distancematrix))
#     emdscoresran.append(emd(norrandom,gtdist,distancematrix))
#     # emdscores2.append(emd(corrs2,gtdist,distancematrix))
#     # emdscores3.append(emd(corrs3,gtdist,distancematrix))
#     # emdscores4.append(emd(corrs4,gtdist,distancematrix))
#     # emdscores12.append(emd(corrs12,gtdist,distancematrix))
#     emdscoressum.append(emd(sumall,gtdist,distancematrix))
#     # emdscores1_2.append(emd(corrs1,corrs2,distancematrix))

# print(np.mean(scores))
# print(np.mean([score1,score2,score3,score4]))
# print(np.mean(scoresran))
# print(np.mean(scores1))
# print(np.mean(scores2))
# print(np.mean(scores3))
# print(np.mean(scores4))
# print(np.mean(scores12))
# print(np.mean(scoressum))
# print(np.mean(scores1_2)) 
# print(np.mean(emdscoores))
# print(np.mean(emdscoresran))
# print(np.mean(emdscores1))
# print(np.mean(emdscores2))
# print(np.mean(emdscores3)) 
# print(np.mean(emdscores4))
# print(np.mean(emdscores12))
# print(np.mean(emdscoressum))
# print(np.mean(emdscores1_2)) 

# print("***********************")
# print(np.var(scores))
# print(np.var(scoresran))
# print(np.var(scores1))
# print(np.var(scores2))
# print(np.var(scores3))
# print(np.var(scores4))
# print(np.var(scores12))
# print(np.var(scoressum))
# print(np.var(scores1_2))
# print(np.var(emdscoores))
# print(np.var(emdscoresran))
# print(np.var(emdscores1))
# print(np.var(emdscoressum))

# # X = np.vstack((corrs, gtdist))
# # Y = pdist(X,'cityblock')
# # print("score="+str(Y))

# # X = np.vstack((norrandom, gtdist))
# # Y = pdist(X, 'cityblock')
# # print("scoreran="+str(Y))
# # X = np.vstack((corrs1, gtdist))
# # Y = pdist(X, 'cityblock')
# # print("score1="+str(Y))

# # X = np.vstack((corrs2, gtdist))
# # Y = pdist(X, 'cityblock')
# # print("score2="+str(Y))

# # X = np.vstack((corrs3, gtdist))
# # Y = pdist(X, 'cityblock')
# # print("score3="+str(Y))

# # X = np.vstack((corrs4, gtdist))
# # Y = pdist(X, 'cityblock')
# # print("score4="+str(Y))

# # X = np.vstack((corrs1, corrs2))
# # Y = pdist(X, 'cityblock')
# # print("score1_2="+str(Y))

# # X = np.vstack((corrs2, corrs3))
# # Y = pdist(X, 'cityblock')
# # print("score2_3="+str(Y))

# # X = np.vstack((corrs3, corrs4))
# # Y = pdist(X, 'cityblock')
# # print("score3_4="+str(Y))

# # X = np.vstack((corrs1, corrs3))
# # Y = pdist(X, 'cityblock')
# # print("score1_3="+str(Y))
#     # scores = [];
#     # for i in range(len(otheralg)):
#     #     score = [];
#     #     for j in range(len(otheralg)):
#     #         if numpy.mean(otheralg[i]) == 0:
#     #             print(i)
#     #         if numpy.mean(otheralg[j]) == 0:
#     #             print(j)

#     #         score.append(bhatta(otheralg[i],otheralg[j]) );
#     #     scores.append(score);

#     # for i in scores:
#     #     print i


#     # plt.matshow(mxall, aspect='auto') 
#     # # , cmap=plt.cm.Greys)
#     # meancorr = np.diff(corrs.mean(axis=0)) 
#     # # medcorr = medfilt(meancorr, kernel_size=5)
#     # # plt.plot(meancorr, color='y')
#     # plt.plot(np.gradient(corrs.mean(axis=0))*100)
#     # plt.plot(np.diff(mxnp.sum(axis=0))*5)
#     # plt.show()