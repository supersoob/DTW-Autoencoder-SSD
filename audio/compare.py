import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from dtw import dtw
#from dtaidistance import dtw
#from fastdtw import fastdtw
from scipy.spatial.distance import correlation
from numpy.linalg import norm
import sys

#Loading audio files
img_num=sys.argv[1]
y1, sr1 = librosa.load(f'sound_img{img_num}_deg0.wav')
dst1=[]
dst2=[]
xrange=[]

for i in range(-20,22,2):
    y2, sr2 = librosa.load(f'sound_img{img_num}_deg{i}.wav')

    if i < 20:
        y3, sr3 = librosa.load(f'sound_img{img_num}_deg{i+2}.wav')

    #Showing multiple plots using subplot
    #plt.subplot(1, 2, 1)

    float_y1 = y1.astype(np.float)
    mono_sound1 = librosa.to_mono(float_y1)
    mfcc1 = librosa.feature.mfcc(mono_sound1, sr1)  # Computing MFCC values
    #librosa.display.specshow(mfcc1)

    #plt.subplot(1, 2, 2)
    float_y2 = y2.astype(np.float)
    mono_sound2 = librosa.to_mono(float_y2)
    mfcc2 = librosa.feature.mfcc(mono_sound2, sr2)
    #librosa.display.specshow(mfcc2, x_axis='time')
    #librosa.display.specshow(mfcc2)
    #print(mono_sound2)

    dist1, cost1, acc_cost1, path1 = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))

    if i < 20:
        #print(i,(i+2))
        y3, sr3 = librosa.load(f'sound_img{img_num}_deg{i+2}.wav')
        float_y3 = y3.astype(np.float)
        mono_sound3 = librosa.to_mono(float_y3)
        mfcc3 = librosa.feature.mfcc(mono_sound3, sr3)  # Computing MFCC values
        dist2, cost2, acc_cost2, path2 = dtw(mfcc2.T, mfcc3.T, dist=lambda x, y: norm(x - y, ord=1))
        dst2.append(dist2)
    #dist = dtw.distance(y1,y2)

    #print("The normalized distance between the two : ",dist)   # 0 for similar audios

    xrange.append(i)
    dst1.append(dist1)


    #plt.imshow(cost.T, origin='lower', cmap=plt.get_cmap('gray'), interpolation='nearest')
    #plt.plot(path[0], path[1], 'w')   #creating plot for DTW
    #plt.xlim((-0.5, cost.shape[0]-0.5))
    #plt.ylim((-0.5, cost.shape[1]-0.5))

    #plt.show()  #To display the plots graphically

#plt.close()
print(len(dst1))
print(len(dst2))
plt.plot(xrange,dst1,xrange[:20],dst2,marker="o")
plt.grid()
#plt.title(f'Test Image {img_num}')
plt.legend(['Similarity','Pair Similarity'])
plt.xlim(-20, 20)
plt.ylim(-3,250)
plt.xlabel('Displacement (Deg)')
plt.ylabel('DTW distance')
plt.savefig(f"graph/test_figure_{img_num}deg.png",dpi=300)


#plt.show()
plt.close()