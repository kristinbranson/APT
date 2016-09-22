### APT
Animal Part Tracker

### Requirements
* MATLAB R2014b or later.
* Windows, Linux, and Mac are all supported, but Windows and Linux get priority.
* A recent checkout of JCtrax or JAABA. Development is being done primarily against JAABA but it's supposed to work with either.

### Usage
1. Copy Manifest.sample.txt to Manifest.txt and edit to point to your checkout of JCtrax or JAABA (specify the root directory, which contains the subfolders filehandling/ and misc/). You can ignore the cameracalib entry unless you are doing multiview labeling/tracking.
2. Open MATLAB and:
```
% in MATLAB
cd /path/to/git/APT/checkout
APT.setpath % configures MATLAB path
lObj = Labeler;
```
3. Go to the File> menu and either "Quick Open" a movie, or create a "New Project". Quick Open will prompt you for a movie file, then a trx file; if you don't have a trx file, just Cancel.

If you have never created a project, "Quick Open" may not give you the appropriate number of labeling points. However, this could still be a useful jumpstart to start getting familiar with the program.

### Description

#### Project Setup (new project configuration)
When creating a new project:

* The **Number of Points** is the number of points you will adjust/set in each frame during labeling.
  * For a multiview project, this is the number of **physical, 3D points**. Currently it is assumed that every physical point appears in every view.
* The **Number of Views** should be set to 1 unless you have multiple cameras/views.
* Notes on some of the available **Labeling Modes** are below. Template Mode is a good starting point if you are new to APT.
* **Tracking**. For CPR tracking, at the moment you will need a checkout of the CPR repository. See also https://github.com/kristinbranson/APT/wiki/CPR-Tracking-in-the-Labeler.
* Under the Advanced pane:
  * Properties under **LabelPointsPlot** specify cosmetics for clicked/labeled points.

After creating a new project, File>Manage Movies lets you add/remove movies to your project, switch the current movie being labeled, etc.

Conceptually, a Project (or Project file) consists of a project configuration, a list of movies (optionally with trx files), labels, and labeling-related metadata.  By default, Projects are saved to files with the .lbl extension.

When you open a movie with "Quick Open", you are actually creating a project and adding a single movie. This Quick-Opened movie is a project like any other.

##### Sequential Mode
Click the image to label. When NumLabelPoints points are clicked, adjust the points with click-drag. Hit accept to save/lock your labels. You can hit Clear at any time, which starts you over for the current frame/target. Switch targets or frames and do more labeling; the Targets and Frames tables are clickable for navigation. See the Help> menu for hotkeys. When browsing labeled (accepted) frames/targets, you can go back into adjustment mode by click-dragging a point. You will need to re-accept to save your changes. When you are all done, File>Save will save your results.

##### Template Mode
The image will have NumLabelPoints white points overlaid; these are the template points. Click-drag to adjust, or select points with number keys and adjust with arrows or mouse-clicks. Points that have been adjusted are colored. See the Help> menu for hotkeys. Click Accept to save/lock your labels. Switch targets/frames and the template will follow.

Both "fully occluded" and "occluded estimate" points are supported. To set a point as an occluded estimate, use right-click, or use the 'o' hotkey when the point is selected. For fully occluded points, select the point and click in the box in the lower-left of the main image as usual.

When working with more than 10 points, use backquote (`) to re-map the 0-9 hotkeys to larger-index points.

##### HighThroughput (HT) Mode
In HT mode, you label the entire movie for point 1, then you label the entire movie for point 2, etc. Click the image to label a point. After clicking/labeling, the movie is automatically advanced NFrameSkip frames. When the end of the movie is reached, the labeling point is incremented, until all labeling for all NumLabelPoints is complete. You may manually change the current labeling point in the Setup>HighThroughput Mode menu.

Right-clicking the current point offers options for repeatedly accepting the current point. Right-clicking the image labels a point as an "occluded estimate", ie the clicked location represents your best guess at an occluded landmark. 

HT mode was initially intended to work on movies with no existing trx file (although this appears to work fine).


#### Occluded
To label a point as "fully occluded", click in the box in the lower-left of the main image. Depending on the mode, you may be able to "unocclude" a point, or you can always push Clear.

High-Throughput and Template modes currently support marking points as "occluded estimates" via right-click or the 'o' hotkey. These labels represent best guesses at an occluded landmark.

Fully occluded points appear as inf in the .labeledpos Labeler property; occluded-estimate points are tagged in .labelpostag.

#### Tracking

The Labeler is designed to work with pluggable "Trackers" which attempt to learn from and predict labeled data.

A version of CPR (Cascaded Pose Regression) has been implemented and is available for use: see
https://github.com/kristinbranson/APT/wiki/CPR-Tracking-in-the-Labeler
