
# coding: utf-8

# In[ ]:

# import cv2
import sys
import os
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
sys.path.append('/home/mayank/work/pyutils')
import myutils

if __name__ == '__main__':
    
    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

    curdir = '/home/mayank/Dropbox/AdamVideos/multiPoint/M122_20140828/M122_20140828_v002'
#     FFMpegWriter = manimation.writers['ffmpeg_file']
    FFMpegWriter = manimation.writers['ffmpeg']
    writer = FFMpegWriter(fps=15,bitrate=-1)

    fig = plt.figure()

    cap = cv2.VideoCapture(os.path.join(curdir,'movie_comb.avi'))
    plt.gray()
    with writer.saving(fig,"test_results_mencoder.mp4",4):
        for fnum in range(0,50):
#             print(fnum)
            plt.clf()
            framein = myutils.readframe(cap,fnum)
            plt.imshow(framein)
            writer.grab_frame()
    cap.release()        


# In[13]:

get_ipython().magic(u'matplotlib inline')
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

curdir = '/home/mayank/Dropbox/AdamVideos/multiPoint/M122_20140828/M122_20140828_v002'
fig = plt.figure()

cap = cv2.VideoCapture(os.path.join(curdir,'movie_comb.avi'))
plt.gray()
for fnum in range(0,2):
    print(fnum)
    plt.clf()
    framein = myutils.readframe(cap,fnum)
    plt.imshow(framein)
cap.release()        


# In[11]:

plt.imshow(framein)
plt.show()

