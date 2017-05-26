## Orthocam Stereo Calibration

This is a step-by-step guide to running an OrthoCam stereo camera calibration. OrthoCam is intended to provide a more usable/stable calibration for cameras/rigs which operate in the weak perspective (delta-z << z) regime.

#### Setup

* Add the APT repo to your MATLAB path, or navigate to the <APT> repo root directory. Then run `APT.setpath` to fully configure your MATLAB path.

#### Step 1: Run single-cam calibrations of each camera in the MATLAB Camera Calibrator App.
The MATLAB Camera Calibrator App should do single-camera calibrations pretty well, except the (z-depth, focal length) parameter pair will not be well-resolved. Check that the calibration looks good (low reprojection error). Then do a "Save Session". This will save the calibration images you used with their detected corners, along with the calibration results.

Run/save single-camera calibrations for both cameras. These will be used by OrthoCam.  You may need to intialize the optimization for this to work (e.g.for Stephen's rig go to 'optimization options' and enter " [100000 0 0;0 100000 0;384 255 1]")

#### Step 2: Open the MATLAB Stereo Camera Calibrator App (stereoCameraCalibrator) and Create a new Stereo Project.
Add the calibration image pairs you would like to use and verify that the corners are detected appropriately etc.

**Important:** *All calibration images included/used in the stereo project need to have been included in the single-camera calibrations in Step 1.*

#### Step 3: Click Calibrate. If your MATLAB path is configured correctly, this will run the OrthoCam stereo calibration instead of the MATLAB calibration.

**Step 3a. Obtain single-camera Orthocam calibrations for each camera.**
  
First, you will be prompted to load your saved single-camera calibration sessions from the MATLAB App, first for camera 1, then camera 2.

Single-camera OrthoCam calibrations will be done on each camera, using the MATLAB results as a guideline/starting point. Currently, the MATLAB optimizer *lsqnonlin* is used to run these optimizations. In the progress display, the third column of numbers indicates the current optimizer residual. A "good" value for this residual is say 1000 or less. This residual may decreases slowly at times but will often make large "quantum leaps".  

Verify that the reproduction error is good with these single-camera calibrations.

After both single Orthocam calibrations are done, you can save the results. The next time you need to run this stereo calibration, you can skip the mono-calibrations and directly load these single-camera OrthoCam calibration results.

**Step 3b. Optionally, select a "base" pattern for the stereo calibration.**
This is a technical quirk with the current implementation. One calibration-pattern-pair is currently selected to serve as a common coordinate system for the optimization. Pattern-pairs that are "unusual" (eg upside-down, far from image center(s) etc) should not be chosen as they can lead to convergence problems. Select a very "normal" pattern-pair for this step  (X to right; Y down).

**Step 3c. Run the stereo optimization.**
Again, *lsqnonlin* is used for the optimization. This optimization is tougher, and usually requires restarts. Rather than just letting the optimizer run continuously for a long time, using restarts seems to shock/randomize the optimizer and provides for a faster overall optimization. The residual will decrease slowly at times and will make large leaps at other times. 

You can restart/repeat the optimization until the final optimizer message is "Possible minimum found" etc rather than "Maximum iterations/evaluations exceeded". Again, a "good" residual value (3rd column of numbers in display) is say in the range of 1000 or less.

Verify that the reproduction error is good with the stereo calibration. 

Verify the extrinsics.

You will be prompted to save your Orthocam calibration.

At this point you are done -- the final results are not integrated with the MATLAB Stereo Camera Calibration App, so you may not see the usual MATLAB App display after completion. 
 

