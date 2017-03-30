## Orthocam Stereo Calibration

This is a step-by-step guide to running an OrthoCam stereo camera calibration. OrthoCam is intended to provide a more usable/stable calibration for cameras/rigs which operate in the weak perspective (delta-z << z) regime.

#### Setup

* Add <APT>/user/orthocam to your MATLAB path. 

#### Step 1: Run single-cam calibrations of each camera in the MATLAB Camera Calibrator App.
The MATLAB App should do single-camera calibrations pretty well, except the (z-depth, focal length) parameter pair will not be well-resolved. Check that the calibration looks good (low reprojection error). Then do a "Save Session". This will save the calibration images you used with their detected corners, along with the calibration results.

#### Step 2: Open the MATLAB Stereo Camera Calibrator App and Create a new Stereo Project.
Add the calibration image pairs you would like to use and verify that the corners are detected appropriately etc.

**Important:** All calibration images included/used here need to have been included in the single-camera calibrations in Step 1.

#### Step 3: Click Calibrate. If your MATLAB path is set up, this will run the OrthoCam stereo calibration instead of the MATLAB calibration.

First, you will be prompted to load your saved single-camera calibration sessions from the MATLAB App, first for camera 1, then camera 2.

Single-camera OrthoCam calibrations will be done on each camera, using the MATLAB results as a guideline/starting point. Currently, the MATLAB optimizer *lsqnonlin* is used to run these optimizations. In the progress display, the third column of numbers indicates the current optimizer residual. A "good" value for this residual is say 1000 or less. This residual may decreases slowly at times but will often make large "quantum leaps".  

Verify that the reproduction error is good with these single-camera calibrations.

After both cameras have been single-camera calibrated, you can optionally save the single-camera calibrations.

Now, the stereo OrthoCam calibration will proceed. Again, *lsqnonlin* is used. This optimization is tougher, and usually requires restarts. (Rather than just letting the optimizer run continuously for a long time, using restarts seems to shock/randomize the optimizer and provides for a faster overall optimization.) Again, the residual will decrease slowly at times and will make large leaps at other times. A "good" value again is say in the range of 1000.

Verify that the reproduction error is good with the stereo calibration. 

Verify the extrinsics.

You can now save the stereo Orthocam calibration. 

