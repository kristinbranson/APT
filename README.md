[Original KB discussion](Link URL)


### TODO/RADAR ###

* Features
    * "I would reorganize ftrsGenDup2, ftrsCompDup2 to be less of a mess. I think there are a few flags we want to set, like whether we only consider neighboring landmarks or not, and whether we only update some landmarks in an iteration. Also, remove some of the obsolete feature types, clean up the indexing, maybe use keywords instead of numbers?"
    * Features are getting clipped due to the radius being ~50 (already in units of pixels) and then this value scaling something else (eg major axis of ellipse) that is in pixels.
    * performance tweak, getLinePoint2. (Re features it's all in the comp rather than gen)
* The MEX-files can use some doc-ing of input/output args etc (in some cases the comments appear wrong, in some cases args might be obsolete)


### State of code 20151112 ###

RCPR originally downloaded from
http://www.vision.caltech.edu/xpburgos/ICCV13/
Modified by JR first, and then by KB -- rather a big mess.

Main scripts from KB:

Locations of scripts are historical :/...

video_tracking/ScriptPrepareTrackingData_*.m: 
Find and parse manually labeled data in various formats, put it into a single mat file containing the following variables:
pts: nlandmarks x ndims x ntrainexamples matrix containing the locations of the landmarks in each frame. 
ts: 1 x ntrainexamples vector, ts(i) is the frame number corresponding to label pts(:,:,i). 
expidx: 1 x ntrainexamples vector, expidx(i) describes which video label pts(:,:,i) corresponds to. 
Depending on the type of data, the locations of the videos are described differently:

Stephen's fly-head data (FlyHeadStephen20150810): 
vid1files: view 1 video
vid2files: view 2 video
matfiles: registration metadata

Adam's mouse paw data (M118to174_20150615), larva data (Larva20150506), fly bubble data (FlyBubble20150502, FlyBubble_Legs20150614):
expdirs: Directories containing each video
moviefilestr: name of video within expdir.

In addition, larva and fly bubble data have:
flies: 1 x ntrainingexamples vecotr, flies(i) describes the fly corresponding to label pts(:,:,i). 

misc/ScriptTrainTracker_*.m:

* Reads in the labeled frames from all videos, stores in a cell. May need to be changed for memory reasons. 
* For the mouse data, we subsample the data because there is a lot. 
* For some types of data, we do histogram equalization to try to get intensities across videos consistent. I played with a few methods for this. In all cases (I think?) I wanted the equalization to be per-video, not per-frame, so I did something stupid, which is created one really big image from all training images in a given video, and do histogram equalization on that. This is also very inefficient, and I was never very happy with how histeq was working. Options are to improve this step or to make the features used later in the algorithm more intensity invariant. 
* save the training data to a file
* For some types of data, we want to know which landmark points are "neighbors" of each other -- are closest in location to each other across the data. See below for more info. 
* Set a lot of parameters
* Train tracker using the function *train.m* (in old versions of the mouse tracker, a sequence of three trackers were trained, one for both paws simultaneously first, then two individual view trackers. This should be obsolete now, but there might be vestiges of this). 
* Various ways of testing whether the tracker is working. Main function for running the tracker on a video: *test.m*. 





Main functions: 

train.m: Wrapper function for main training function from original RCPR code, rcprTrain.m. 

Inputs:

* phisTr: ntrainexamples x (nlandmarks\*ndims) matrix containing the landmark locations (basically a permuted version of labels.pts from above). 
* bboxesTr: ntrainexamples x 4 matrix defining the search region. CPR only uses normalized landmark positions in some coordinate system. This defines the initial coordinate system. Almost always, this is just the whole image. 
* IsTr: 1 x ntrainexamples cell containing the images corresponding to the labeled landmarks. In all our data, these images are the same size, but the code is currently general enough to allow different image sizes. This could potenially be made less general to make the feature grabbing faster. 
* Lots of model parameters. Here are the important ones:
    * cpr_type: All of our experiments have been with cpr_type = 2. This means that we are using the robust casdaded pose regression approach from the Burgos-Artizzu paper, but not the occlusion modeling stuff. We need to get occlusion modeling working at some point. 
    * feature_type: We have added a bunch of different feature types to the code to try different ways of generating features. 
These were added by JR:

  5: Select any pair of landmarks, select a feature within the ellipse aligned with the landmarks with semimajor-axis length radius*d and semiminor-axis length radius*d/2, where d is the distance between the two landmarks (just realized it was doing this stretching in one particular direction!), and radius is a parameter. 
  6: Select one landmark, select a feature from same radius around that landmark. 
  7: ?? obsolete I think. 
  8: Same as type 5, but only consider pairs of landmarks that are defined as neigbors. So: select a pair of *neighboring* landmarks, select a feature within the ellipse aligned with the landmarks with semimajor-axis length d and semiminor-axis length d/2, where d is the distance between the two landmarks. 
  9: Same as type 5, but instead of choosing a feature in an ellipse around the mid-point of the two landmarks, selects a point around a randomly selected point between the two landmarks. This probably doesn't make a lot of sense now -- I didn't realize that it was doing this axis-aligned ellipse thing. My goal was just to select a feature within some distance of the line segment connecting the two landmarks. So: select a point along the line segment connecting a pair of landmarks, select a feature within an axis-aligned ellipse centered on that point. 
  10: Same as type 9, but only consider a subset of the landmarks. So: select a point along the line segment connecting a pair of landmarks from the set "fids" (fids is selected per iteration), select a feature within an axis-aligned ellipse centered on that point. 
  11: Same as type 9, but only consider pairs of landmarks that are defined as neighbors. So: select a point along the line segment connecting a pair of landmarks defined as neighbors, select a feature within an axis-aligned ellipse centered on that point.