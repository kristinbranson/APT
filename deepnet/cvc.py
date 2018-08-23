
# coding: utf-8

# In[1]:

# constant for different versions of opencv
from builtins import object
import cv2
class cvc(object):
    x = cv2.__version__.split('.')
    major = x[0]
    if major == '2':
        FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT
        FRAME_WIDTH = cv2.cv.CV_CAP_PROP_FRAME_WIDTH
        FRAME_HEIGHT = cv2.cv.CV_CAP_PROP_FRAME_HEIGHT
        FRAME_POSITION = cv2.cv.CV_CAP_PROP_POS_FRAMES
        FRAME_MSEC = cv2.cv.CV_CAP_PROP_POS_MSEC
        FRAME_FPS = cv2.cv.CV_CAP_PROP_FPS
    else:
        FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
        FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
        FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
        FRAME_POSITION = cv2.CAP_PROP_POS_FRAMES
        FRAME_MSEC = cv2.CAP_PROP_POS_MSEC
        FRAME_FPS = cv2.CAP_PROP_FPS

    

