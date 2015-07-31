#### APT
Animal Part Tracker

This is a prototype!

#### Requirements
MATLAB R2014b or later. Development is being done on Win7 with some testing on Linux.

#### Usage
```
% in MATLAB
cd /path/to/git/APT/checkout % root of checkout contains aptroot.m
setaptpath 
lObj = Labeler(LabelMode.SEQUENTIAL,5); % Open Labeler in Sequential mode with 5-point models
lObj = Labeler(LabelMode.TEMPLATE,5); % Open Labeler in Template mode with 5-point models

```

Go to the File> menu to open a movie or movie+trx. 

#### Description

###### Sequential Mode
Click the image to label. When 5 points are clicked, you can adjust the points with click-drag. Hit accept to save/lock your labels. You can hit Clear at any time, which starts you over for the current frame/target. Switch targets or frames and do more labeling; the Targets and Frames tables are clickable for navigation. See the Help> menu for barebones hotkeys. When browsing labeled (accepted) frames/targets, you can go back into adjustment mode by click-dragging a point. You will need to re-accept to save your changes. When you are all done, File>Save will save your results.

###### Template Mode
The image will have 5 white points overlaid; these are the template points. Click-drag to adjust, or select points with number keys and adjust with arrows or mouse-clicks. See Help> menu for hotkeys. Click Accept to save/lock your labels. Switch targets/frames and the template will follow. 

