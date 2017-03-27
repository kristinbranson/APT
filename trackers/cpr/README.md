## CPR

Cascaded Pose Regression tracker

## Requirements

* A recent version of MATLAB (R2014b or later preferred) with Image Processing, Stats toolboxes
* A recent checkout of JCtrax (https://github.com/kristinbranson/JCtrax) **or** JAABA (https://github.com/kristinbranson/JAABA)
* A recent checkout of Piotr Dollar's toolbox (https://github.com/pdollar/toolbox)
* If running tracking with the Labeler (APT), a recent checkout of APT (https://github.com/kristinbranson/APT)

## Setup

Copy Manifest.sample.txt to Manifest.txt and edit to point to your local checkouts. 

## Usage

In MATLAB, instead of APT.setpath, you can run:

    ```
    % in MATLAB
    cd /path/to/CPR/checkout
    CPR.setpath % configures MATLAB path
    lObj = Labeler;
    ```

This will configure your CPR path, which will include your APT path.

At the moment the only documented usage is through the APT Track/Train interface.