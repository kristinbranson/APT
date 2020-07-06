### APT: The Animal Part Tracker

![APT Examples](https://github.com/kristinbranson/APT/blob/master/docs/images/apt_examples.png)

APT is a machine-learning based software package that enables tracking the pose or posture of behaving animals in video. APT can work with potentially any animal (or animals), in any setting or experimental configuration. Its major functionality includes:

  * Implementations of a number of leading DNN (deep neural network) architectures for learning and prediction, as well as extensibility to new, user-defined deep networks
  * Support for GPU training and tracking on a local workstation with our [Docker image](https://github.com/kristinbranson/APT/wiki/Linux-&-Docker-Setup-Instructions), in the AWS cloud, or with the JRC GPU cluster (for Janelians) 
  * A fully-featured graphical interface 
  * A rich MATLAB command-line API for scripting and advanced users
  * Support for multi-camera data with 3D-enabled labeling and tracking
  * Support for projects with multiple animals and/or externally-generated body tracking

... and much more!

### User Guide

A basic, preliminary user guide with installation and setup instructions can be found at [http://kristinbranson.github.io/APT](http://kristinbranson.github.io/APT/). 

More recent or advanced documentation can be found in the [wikis](https://github.com/kristinbranson/APT/wiki). This wiki is sorted chronologically with the most recent updates at the top.

### Contributors
APT is being developed in the Branson lab by Allen Lee, Mayank Kabra, Kristin Branson, Alice Robie, and Roian Egnor, with help from many others. All work is funded by the Howard Hughes Medical Institute and the Janelia Research Campus. APT is currently under heavy development. Please contact Kristin Branson if you are interested in using it.

### License
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [GNU General Public License (version 3)](http://GNU_GPL_v3.html) for more details.

### Sources

APT contains code from the following sources:
* deepnet/deepcut:
  https://github.com/AlexEMG/DeepLabCut
  A. & M. Mathis Labs
  DeepLabCut2.0 Toolbox (deeplabcut.org) 
* deepnet/leap
  https://github.com/talmo/leap
  Talmo Pereira: talmo(at)princeton.edu
* deepnet/imagenet_resnet.py
  https://github.com/jgraving/deepposekit
  Jacob Graving, Daniel Chae, Hemal Naik, Liang Li, Benjamin Koger, Blair Costelloe, and Iain Couzin
* matlab/trackers/cpr
  http://www.vision.caltech.edu/xpburgos/ICCV13/code/rcpr_v1.zip
  X.P. Burgos-Artizzu, P.Perona, and Piotr Dollar 
* matlab/javaaddpathstatic.m:
  http://stackoverflow.com/questions/19625073/how-to-run-clojure-from-matlab/22524112#22524112
  Andrew Janke
* matlab/JavaTableWrapper
  https://www.mathworks.com/matlabcentral/fileexchange/49994-java-table-wrapper-for-user-interfaces
  Robyn Jackey
* matlab/jsonlab-1.2
  https://www.mathworks.com/matlabcentral/fileexchange/33381-jsonlab-a-toolbox-to-encode-decode-json-files
  Qianqian Fang
* matlab/propertiesGUI
  http://undocumentedmatlab.com/articles/propertiesgui
  Yair M. Altman: altmany(at)gmail.com
* matlab/treeTable
  http://undocumentedmatlab.com/articles/treetable
  Yair M. Altman: altmany(at)gmail.com
* matlab/YAMLMatlab_0.4.3
  https://code.google.com/archive/p/yamlmatlab/
  Jiri Cigler, Jan Siroky, Pavel Tomasko
* matlab/misc/ellipsedraw
  https://www.mathworks.com/matlabcentral/fileexchange/3224-ellipsedraw1-0
  Lei Wang
* matlab/misc/findjobj_modern.m
  https://www.mathworks.com/matlabcentral/fileexchange/14317-findjobj-find-java-handles-of-matlab-graphic-objects
  Yair Altman
* matlab/misc/glob.m
  https://www.mathworks.com/matlabcentral/fileexchange/40149-expand-wildcards-for-files-and-directory-names
  Peter van den Biggelaar
* matlab/misc/saveJSONfile.m
  https://www.mathworks.com/matlabcentral/fileexchange/50965-structure-to-json
  Lior Kirsch
* matlab/misc/whereisjavaclassloadingfrom.m
  https://stackoverflow.com/questions/4376565/java-jpa-class-for-matlab/4380622#4380622
  Andrew Janke
* matlab/user/APT2RT
  http://www.mathworks.com/matlabcentral/fileexchange/22422-absolute-orientation
  Christian Wengert
  http://www.mathworks.com/matlabcentral/fileexchange/35475-quaternions
  Przemyslaw Baranski
  https://www.mathworks.com/matlabcentral/fileexchange/24484-geom3d
  David Legland
  https://isbweb.org/software/movanal/kinemat/
  Christoph Reinschmidt, Ton van den Bogert
* external/JAABA
  http://jaaba.sourceforge.net/
  Mayank Kabra, Kristin Branson, et al.
* external/CameraCalibrationToolbox
  http://www.vision.caltech.edu/bouguetj/calib_doc/
  Jean-Yves Bouguet
* external/netlab
  https://www.mathworks.com/matlabcentral/fileexchange/2654-netlab
  Ian T. Nabney
* external/PiotrDollarToolbox
  https://pdollar.github.io/toolbox/
  Piotr Dollar
* matlab/misc/inputdlgWithBrowse.m (original: inputdlg.m)
  matlab/misc/imcontrast_kb.m (original: imcontrast.m)
  matlab/private_imuitools/\*.m (requirements for imcontrast.m)
  matlab/user/orthocam/\*
  Modified MATLAB code.


