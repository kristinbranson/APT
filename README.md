## APT
Animal Part Tracker

## Requirements
* MATLAB R2014b or later.
* Windows, Linux, and Mac are all supported, but Windows and Linux get priority.
* A recent checkout of [JAABA](https://github.com/kristinbranson/JAABA).
* If you are working with multi-camera data, you may need calibration software for your rig, eg
  * [Caltech camera calibration toolbox](https://www.vision.caltech.edu/bouguetj/calib_doc/)

## Setup
1. Copy <APT>\Manifest.sample.txt to Manifest.txt and edit to point to your checkout of JAABA (specify the root directory, which contains the subfolders filehandling/ and misc/). 
2. For multi-camera data, edit the 'cameracalib' entry to point to your calibration software.
2. Open MATLAB and:

    ```
    % in MATLAB
    cd /path/to/git/APT/checkout
    APT.setpath % configures MATLAB path for running APT
    lObj = Labeler;
    ```
    
3. Go to the File> menu and either "Quick Open" a movie, or create a "New Project". Quick Open will prompt you for a movie file, then a trx file; if you don't have a trx file, just Cancel.

If you have never created a project before, "Quick Open" may not give you the appropriate number of labeling points. However, this could still be a useful jumpstart to start getting familiar with the program.

## Usage

### New Project Setup
When creating a new project:

* The **Number of Points** is the number of points you will adjust/set in each frame during labeling.
  * For a multi-camera project, this is the number of **physical, 3D points**. Currently it is assumed that every physical point appears in every view.
* The **Number of Views** should be set to the number of cameras/views for your data. Typically this will be 1.
* Notes on some of the available **Labeling Modes** are below. **Template Mode** is a good starting point if you are new to APT. For multi-camera projects, **Multiview Calibrated 2** is most popular.
* **Tracking**. For CPR tracking, at the moment you will need a checkout of the CPR repository. See also https://github.com/kristinbranson/APT/wiki/CPR-Tracking-in-the-Labeler.
* Under the Advanced pane:
  * Properties under **LabelPointsPlot** specify cosmetics for clicked/labeled points.

After creating a new project, File>Manage Movies lets you add/remove movies to your project, switch the current movie being labeled, etc.

Conceptually, a Project (or Project file) consists of a project configuration, a list of movies (optionally with trx files), labels, and labeling-related metadata.  By default, Projects are saved to files with the .lbl extension.

When you open a movie with "Quick Open", you are actually creating a project and adding a single movie. This Quick-Opened movie is a project like any other.

### Adding Movies 

After creating a project, the "Manage Movies" window should appear. This window lets you add, remove and select movies.

For multi-camera projects, the **Add Movie** button actually adds a **movieset**, a group of associated movies, one for each camera/view. When prompted, select all movies for a new movieset at once. The ordering here is important; make sure you are consistent about the ordering of cameras/views in moviesets. (During project setup, you can optionally name the views, eg View1="Front" and View2="Side".) 

### Labeling Modes

For all labeling modes, the Help> menu contains some useful tips.

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

##### Multiview Calibrated 2
For calibrated labeling, first go to Setup> Load Calibration File to load a calibration file. This enables calibrated labeling, where epipolar lines and reconstructed points are shown to assist in labeling multi-camera data. 

The number keys 0-9 select a physical point. Each view has its own projection of this point and the selected point will be indicated on all views. After a point is selected, clicking on any view will jump the point in that view to the clicked location. Epipolar lines will be projected in the other views. The point in the first/original view is now adjustable by click-dragging, or with the arrow or <shift>-arrow keys. The epipolar lines should live-update as the first point is adjusted.

For projects with three or more views: Clicking on a second view jumps the selected point in  the second view to the clicked location. With the point **anchored** in two views, reconstructed best-guesses for the point are shown in all remaining views. The reconstruction shows three points indicating a spread of possible locations based on the first two clicked locations. The middle point is the most likely location. 

Spacebar toggles the anchoring state of a point in a view. Anchored points have the letter 'a' appended to their text label. When a point is anchored, epipolar lines or reconstructed points are shown in the other views. Points that are shown as 'x' rather than '+' are adjustable with the arrow keys (fine adjustment) or <shift>-arrow keys (coarse adjustment).   

When all points are in their desired locations, the Accept button (or 's' hotkey) accepts the labels.

#### Occluded
To label a point as "fully occluded", click in the box in the lower-left of the main image. Depending on the mode, you may be able to "unocclude" a point, or you can always push Clear.

High-Throughput and Template modes currently support marking points as "occluded estimates" via right-click or the 'o' hotkey. These labels represent best guesses at an occluded landmark.

Fully occluded points appear as inf in the .labeledpos Labeler property; occluded-estimate points are tagged in .labelpostag.

#### Tracking

The Labeler is designed to work with pluggable "Trackers" which attempt to learn from and predict labeled data.

A version of CPR (Cascaded Pose Regression) has been implemented and is available for use: see
https://github.com/kristinbranson/APT/wiki/CPR-Tracking-in-the-Labeler
