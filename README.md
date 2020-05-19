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
* matlab/javaaddpathstatic.m:
  http://stackoverflow.com/questions/19625073/how-to-run-clojure-from-matlab/22524112#22524112
  Andrew Janke
* matlab/propertiesGUI
  http://undocumentedmatlab.com/articles/propertiesgui
  Yair M. Altman: altmany(at)gmail.com
* matlab/JavaTableWrapper
  https://www.mathworks.com/matlabcentral/fileexchange/49994-java-table-wrapper-for-user-interfaces
  Robyn Jackey
* matlab/private_imuitools
  MATLAB 2011
* matlab/treeTable
  http://undocumentedmatlab.com/articles/treetable
  Yair M. Altman: altmany(at)gmail.com
* matlab/YAMLMatlab_0.4.3
  https://code.google.com/archive/p/yamlmatlab/
  Jiri Cigler, Jan Siroky, Pavel Tomasko
* matlab/misc/saveJSONfile.m
  https://www.mathworks.com/matlabcentral/fileexchange/50965-structure-to-json
  Lior Kirsch
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


