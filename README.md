#### APT
Animal Part Tracker

#### Requirements
* MATLAB R2014b or later. Development is being done primarily on Win7 with light testing on Linux and Mac.
* A fairly up-to-date checkout of JCtrax or JAABA. Development is being done primarily with https://github.com/kristinbranson/JCtrax/commit/1741696319e3900abc3ff2f9c79c1ed185018e51.

#### Usage
1. Copy Manifest.sample.txt to Manifest.txt and edit to point to your checkout of JCtrax or JAABA (specify the root directory, which contains the subfolders filehandling/ and misc/). 
2. Copy pref.default.yaml to pref.yaml and edit for your desired configuration (see below).  
3. Open MATLAB and:
```
% in MATLAB
cd /path/to/git/APT/checkout
APT.setpath % configures MATLAB path
lObj = Labeler;
```
4. Go to the File> menu to open a movie or movie+trx. 

#### Description

###### Configuration/Preferences: pref.yaml
Edit the file /\<APTRoot\>/pref.yaml to set up your labeler; these are your **local settings**. Settings in this file override the default values in /\<APTRoot\>/pref.default.yaml. Your local settings can be a very small subset of pref.default.yaml, for example just a single line. Most importantly:

* **LabelMode** specifies your labeling mode; see LabelMode.m for options. The labeling mode is also automatically set when loading an existing project.
* **NumLabelPoints**.
* **LabelPointsPlot**. Items under this parent specify cosmetics for clicked/labeled points. For HighThroughputMode, **NFrameSkip** specifies the frame-skip. 

###### Sequential Mode
Click the image to label. When NumLabelPoints points are clicked, adjust the points with click-drag. Hit accept to save/lock your labels. You can hit Clear at any time, which starts you over for the current frame/target. Switch targets or frames and do more labeling; the Targets and Frames tables are clickable for navigation. See the Help> menu for hotkeys. When browsing labeled (accepted) frames/targets, you can go back into adjustment mode by click-dragging a point. You will need to re-accept to save your changes. When you are all done, File>Save will save your results.

###### Template Mode
The image will have NumLabelPoints white points overlaid; these are the template points. Click-drag to adjust, or select points with number keys and adjust with arrows or mouse-clicks. Points that have been adjusted are colored. See the Help> menu for hotkeys. Click Accept to save/lock your labels. Switch targets/frames and the template will follow.

Both "fully occluded" and "occluded estimate" points are supported. To set a point as an occluded estimate, use right-click, or use the 'o' hotkey when the point is selected. For fully occluded points, select the point and click in the box in the lower-left of the main image as usual.

When working with more than 10 points, use backquote (`) to re-map the 0-9 hotkeys to larger-index points.

###### HighThroughput (HT) Mode
In HT mode, you label the entire movie for point 1, then you label the entire movie for point 2, etc. Click the image to label a point. After clicking/labeling, the movie is automatically advanced NFrameSkip frames. When the end of the movie is reached, the labeling point is incremented, until all labeling for all NumLabelPoints is complete. You may manually change the current labeling point in the Setup>HighThroughput Mode menu.

Right-clicking the current point offers options for repeatedly accepting the current point. Right-clicking the image labels a point as an "occluded estimate", ie the clicked location represents your best guess at an occluded landmark. 

HT mode was initially intended to work on movies with no existing trx file (although this appears to work fine).

###### Projects
For labeling single movies (with or without trx), use the File>Quick Open Movie menu option. This prompts you to find a moviefile and (optional) trxfile.

When you open a movie in this way, you are actually creating a new Project with a single movie. When you are done labeling or reach a waypoint, you may save your work via the File>Save Project or File>Save Project As.

Conceptually, a Project is just a list of movies (optionally with trx files), their labels, and labeling-related metadata (such as the labeling mode in use). The File>Manage Movies menu lets you add/remove movies to your project, switch the current movie being labeled, etc. By default Projects are saved to files with the .lbl extension.

###### Occluded
To label a point as "fully occluded", click in the box in the lower-left of the main image. Depending on the mode, you may be able to "unocclude" a point, or you can always push Clear.

High-Throughput and Template modes currently support marking points as "occluded estimates" via right-click or the 'o' hotkey. These labels represent best guesses at an occluded landmark.

Fully occluded points appear as inf in the .labeledpos Labeler property; occluded-estimate points are tagged in .labelpostag.

###### Suspiciousness
The Labeler currently allows an externally-computed scalar statistic to be associated with each movie/frame/target. For example one may compute a "suspiciousness" parameter based on the tracking of targets in a movie. The suspiciousness can then be used for navigation within the Labeler (and in the future, notes/curation etc).

Currently, suspiciousness statistics are externally computed and set on the Labeler using **setSuspScore**:
    
``` 
...
lObj = Labeler;
% Open a movie, do some labeling etc
ss = rand(lObj.nframes,lObj.nTargets); % generate random suspiciousness score (demo only)
ss(~lObj.frm2trx) = nan; % set suspScore to NaN where trx do not exist
lObj.setSuspScore({ss}); % set/apply suspScore (cell array because we are setting score for 1st/only movie) 
```

Once a suspScore is set, navigation via the Suspiciousness table is enabled. This table can be sorted by column, but for large movies the performance can be painful.

Note, the Labeler property **.labeledpos** contains all labels for the current project, if these are needed for computing the Suspiciousness. 

###### (Re)Tracking trajectories
(Re)Tracking functionality is currently designed for an iterative workflow where labels are used to update/refine trajectories generated by an external tracker and those new trajectories are used to refine/add labels etc.

To use this functionality, you must first create a Tracker object, which is any handle object with a track() method. For an example, see ToyTracker.m; your track() method must accept the same signature. The track() method accepts the current trajectories and labels, and computes new trajectories.

Tracking is still preliminary, with the current prototype design supporting the following workflow. Consider one of AH's MouseReach movies, which does not have associated trajectories, and suppose we want to track each paw (say in the side view), creating a .trx file with two trajectories, with help from the Labeler.

1. Given an AH MouseReach movie movie_comb.001, create an initial trajectory file paws.trx (save it anywhere) containing two trx elements with start/endframes set appropriately for movie_comb.001. TrxUtil/createSimpleTrx and TrxUtil/initStationary could be useful for this.
2. Configure the Labeler for Sequential-mode labeling, with 2 label points.
3. Start the Labeler, and open movie_comb.001 with paws.trx.
4. Create/set the ToyTracker object with 'trker = ToyTracker; lObj.setTracker(trker);'.
5. This is a little quirky: when Labeling, ignore the fact that there are two trajectories/targets. Instead, leave the first target selected. Use point 1 for the first paw and point 2 for the second paw. 
6. Label a few points. At a minimum, make sure to label the first and last frame of the movie (the ToyTracker clips the trx to start/end at the first/last labeled frames).
7. The Track>Retrack menu item will call the ToyTracker with your labels. The ToyTracker just does a linear interpolation between all labels. The new/resulting trx is then set on the Labeler.
8. Repeat steps 5-7.
9. To save your trx, use Track>Save Trx.


