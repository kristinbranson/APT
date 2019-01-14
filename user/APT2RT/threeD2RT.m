function [translations, axisAngleDegXYZ, quaternions, frameStore,...
    residualErrors, scaleErrors, refHeadOutput, rawXYZcoordsStore, alignedXYZcoordsStore, EulerAnglesDeg_XYZ]...
    = threeD2RT(headData,bodyData,pivotPoint,bodyAngleFrame,frameRate, plotYN, referenceHead)
  

%
% Takes output of Mayank/Kristin's APT head tracking at mulitple frames using 'flyhead' model and calulates
% head rotation (roll, pitch, yaw) each successive digitized frame.
%
% headData = Kine data style formated 3D coordinates of the head markers (antennae tips and bases, centre of cuticle roof over proboscis)
% during trial.  e.g headData.kine.flyhead.data.coords(i_LantTip,1:3,:) = xyz
% coords of left antenna tip for all frames
%
% bodyData = body axis data in old kine format (reformat if using APT data)
% - bodyData.kine.flyhead.config.points.   bodyData.kine.flyhead.data.coords(i_neckCentre,:,bodyAngleFrame);...  
%     bodyData.kine.flyhead.data.coords(i_thoracicAbdomenJoint,:,bodyAngleFrame);... 
%     bodyData.kine.flyhead.data.coords(i_LwingBase,:,bodyAngleFrame);... 
%     bodyData.kine.flyhead.data.coords(i_RwingBase,:,bodyAngleFrame)];
%       Unaligned.  Alignement will be done here
% 
% bodyangleframe = frame number within headDataFname file that contains
% digitized data giving body axis. 
%
%
% pivotPoint = 3 element vector giving best estimate of head pivot point in
% same coords as headDataFname.  Use estimatePivot.m to generate this
% number.  Unaligned.  alignment will be done here.
%
% frameRate = frame rate that video was taken at in frames/sec.
%
% plotYN = Set to 1 if want to plot outputs, 0 if not.
%
% referenceHead =  User provided reference head
% data that all positions will be calculated relative to. In standard format: columns = x;y;z, rows =
% LantTip,RantTip,LantBase,RantBase,ProboscisRoof. Unaligned - alignement
% will be done inside this function
%
% Outputs:  
%
%            
%           
%            EulerAnglesDeg_XYZ = Euler Angles describing rotation from 1st
%            frame in trial to current frame.  Axis of rotation are
%            lab-fixed and are aligned with the tethered fly body.  Y-axis
%            is long body axis, X is left to right for fly and Z is up
%            down.  Euler angles are in degrees and saved in XYZ (pitch, roll, yaw) order.
%            Euler angles were calcualted using the 'ZYX' convention i.e.
%            you apply Z rotation first, then Y, then X to get correct
%            attitude/position.
%
%            translations = translation vector describing translation
%            component of movement between first frame and each subsequent
%            frame.  Units are the same as those you did DLT calibration
%            in.
%
%            AngularVelXYZ_degPerSec = Angular velocities calculated from
%            derivative of Quaternion sequence describing rotation between 1st
%            frame and each subsequent frame.  Units = degrees/sec.  Saved
%            in order of rotation about X, Y and Z axes respectively.
%            X-axis is fly's left-right axis, Y axis is along body long
%            axis, Z is up-down
% 
%            timeaxisMsec = time axis for data in msec
% 
%            stimulusTimeMsec = 2 element vector specifying time in msec
%            that stimulus turned on (should be t=0) and off.
%
%            axisAngleDegXYZ = Rotation between 1st and each subsequent
%            frame expressed in axis-angle format. Each frames data is
%            ordered as [angle in deg, x,y,z coords of axis]
% 
%            quaternions = = Rotation between 1st and each subsequent
%            frame expressed in unit quaternions
% 
%            frameStore = frames each data point corresponds to, zero =
%            frame at which stimulus turned on.
%            
%            residualErrors = Remaining errors once translation and
%            rotation have been accounted for between current frame and 1st
%            frame.  Units are same as those you did the 3D calibration/DLT
%            in.
% 
%            scaleErrors = Mulitplier that had to scale points by to make
%            them fit between first frame and current frame.  Should be
%            small, large values indicate potential error.
%
%            refHeadOutput = head model used to calculate head rotations relative
%            to
%
% Dependencies:
%
%  http://www.mathworks.com/matlabcentral/fileexchange/22422-absolute-orientation
%  http://www.mathworks.com/matlabcentral/fileexchange/35475-quaternions
%  http://www.mathworks.com/matlabcentral/fileexchange/24484-geom3d
%  sjh_rad2deg (mine - easy to replicate if you don't have - just .*180/pi)
%  quat2eulzyx.m (Kine/kine_math)
%
%
% I keep forgetting how this code works, so here are my rough notes from debugging:
% 
% if use artificial ref frame
% 
% 
% labFrameReferenceHeadCoords = artificial reference head used in quaternion calc
% 
% labFrameReferenceHeadCoords currently set to = alignedExampleHeadCoords rotated by 
% rot matrix that rotates headCoords_inModelHeadFormat to be as close as possible to modelheadinlabcoords  (alignedExampleHeadCoords before rotation is made in same way as when just using first frame head coords as ref - align first frame to y-axis)
% 
% headCoords_inModelHeadFormat made by 
% 
% modelheadinlabcoords made by taking various measurements from alignedExampleHeadCoords for length of today parts and then using them to make symmetric frame centered around zero
% 
%     modelheadinlabcoords = [ b2bL/-2,  p2fc,   fc2bc;...%Left base
%                              b2bL/2,   p2fc,   fc2bc;...%right base
%                              0,        p2fc,   fc2bc;...% ant base centroid
%                              0,        p2fc,   0;...%face centroid
%                             ];
% 
% 
% 
%     headCoords_inModelHeadFormat made by just cutting out chunks of alignedExampleHeadCoords that should correspond to frame made in modelheadinlabcoords
% 
%     headCoords_inModelHeadFormat=[alignedExampleHeadCoords(3:4,:);... % 2 ant bases
%                 baseCentroid;... %ant base centroid
%                 faceCentroid;...%face centroid
%                ];
% 
% 
% 
% 
% So in summary: cut out chunk of aligned first frame head to make simple frame, use measurements from same first frame head to make perfectly symmetrical head frame aligned to y-axis. Calculate rotation matrix needed to align these two, then apply it to aligned first frame head to rotate and shift it to be aligned to y-axis making labFrameReferenceHeadCoords that is used to compare data to when making quaternions
% 
% test fly 178
% 
% checked pivot point and is always off resulting in there being translation between aligned head and model head, but always off in a different direction i.e. best compromize
% 
% if use first frame data
% 
% alignedExampleHeadCoords = first frame reference head
% 
% comes from first output of alignBodyAxis2Xaxis() applied to first frame of data
% 
% 




%% general parameters


%If set to 1, generates plots to check for errors.  Otherwise set to 0
debugPlotYN =0;



% indices for different head points - change to autodetect once have text
% IDs in data structure
i_LantTip =  1;
i_RantTip = 2;
i_LantBase =  3;
i_RantBase =  4;
i_ProboscisRoof =  5;

%getting kine data structure indices for different body axis and head pivot points
try %try autodetecting from kine labels
    i_headPivot =  find(ismember(bodyData.kine.flyhead.config.points,'headPivot'));
    i_neckCentre =  find(ismember(bodyData.kine.flyhead.config.points,'neckCentre'));
    i_thoracicAbdomenJoint =  find(ismember(bodyData.kine.flyhead.config.points,'thoracicAbdomenJoint'));
    i_LwingBase =  find(ismember(bodyData.kine.flyhead.config.points,'LwingBase'));
    i_RwingBase =  find(ismember(bodyData.kine.flyhead.config.points,'RwingBase'));
catch %if autodetect from labels didn't work, probably generated by APT - use default values until APT has labels
%    warning('add autodetect of body points indices here once APT has text labels for points')
%    warning('Don''t have APT indices for different body axis points yet')
    i_headPivot = 6;
    i_neckCentre = 7;
    i_thoracicAbdomenJoint = 8;
    i_LwingBase = 9;
    i_RwingBase = 10;
end





%% extracting body axis kine format



bodyCoords = ...
[   bodyData.kine.flyhead.data.coords(i_neckCentre,:,bodyAngleFrame);...  
    bodyData.kine.flyhead.data.coords(i_thoracicAbdomenJoint,:,bodyAngleFrame);... 
    bodyData.kine.flyhead.data.coords(i_LwingBase,:,bodyAngleFrame);... 
    bodyData.kine.flyhead.data.coords(i_RwingBase,:,bodyAngleFrame)];





% Rotating example head data so body axis is aligned with Y axis and wing bases lie along x axis and pivot point is at origin
[ alignedRefHead, ~, ~] = alignBodyAxis2Xaxis( referenceHead,bodyCoords,pivotPoint);




%% for each frame finding quaternion and translation vector that describes 
%rotation/translation between head aligned to body/lab frame and current
%position

%for each frame 
quaternionStore = [];
translationStore = [];
scaleStore = [];
residualStore = [];
frameStore = [];
axisAngleStore_DEGXYZ =[];
LantTipStore = [];
EulerAnglesDeg_XYZ = [];
for n=1:1:size(headData.kine.flyhead.data.coords(i_LantTip,1,:), 3) %for each frame  
    
    %extracting data
    headCoords_inModelHeadFormat = ...
    [   headData.kine.flyhead.data.coords(i_LantTip,:,n);...
        headData.kine.flyhead.data.coords(i_RantTip,:,n);...
        headData.kine.flyhead.data.coords(i_LantBase,:,n);...
        headData.kine.flyhead.data.coords(i_RantBase,:,n);...
        headData.kine.flyhead.data.coords(i_ProboscisRoof,:,n) ];



    % Rotating all data so body axis is aligned with Y axis and wing bases lie along x axis and pivot point is at origin
    [alignedheadCoords, alignedBodyCoords, alignedPivotPoint] = alignBodyAxis2Xaxis(headCoords_inModelHeadFormat,bodyCoords,pivotPoint);

    
    if ~any(any(isnan(alignedheadCoords)))
        %getting rotation matri and translation between 'fake' head aligned with body/lab coordinates current frame and
        %
        %old - [D2M_scale, D2M_R, D2M_T, D2M_residuals]=absoluteOrientationQuaternion(alignedRefHead',alignedheadCoords');
        [q,trans,scaleFac,residuals,rotMat,axisAngleRad,EulerRPY]=computeRotations(alignedRefHead,alignedheadCoords);
    else
        D2M_scale=NaN; 
        D2M_R=nan(3,3);
        D2M_T=nan(3,1);
        D2M_residuals=NaN;
        estRotAxis=nan(1,6);
        estRotAngRad=NaN;
        estRotAng=NaN;
        q=nan(1,4);
    end

    
    
    %storing data as sequence of quaternions and translation vectors and aligned 3D coords with
    %residuals and scale factors as measure of error
    quaternionStore = [quaternionStore;q'];
    translationStore = [translationStore;trans'];
    alignedXYZcoordsStore{n} = alignedheadCoords;
    rawXYZcoordsStore{n} = headCoords_inModelHeadFormat;
    LantTipStore = [LantTipStore;alignedheadCoords(1,:)];%just for convinence during debugging storing one 3D point on own - left antenna base
    axisAngleStore_DEGXYZ =[axisAngleStore_DEGXYZ;axisAngleRad];%axis angle representation of rotation from head aligned with body to current frame - 1st element is angle in degrees, 2:4th elements are xyz rotation axis
    scaleStore = [scaleStore;scaleFac];
    residualStore = [residualStore;residuals];
    EulerAnglesDeg_XYZ = [EulerAnglesDeg_XYZ;EulerRPY];
    frameStore = [frameStore;n];


end






%% formatting for output

translations = translationStore;
interFrameTimeSec = 1/frameRate;
residualErrors = residualStore;
scaleErrors = scaleStore;
axisAngleDegXYZ = axisAngleStore_DEGXYZ;
quaternions = quaternionStore;
refHeadOutput = referenceHead;

%% useful plots for debugging

%debug plot of 3D head positions over time to check angular velocities look
%correct
if debugPlotYN==1
    
    %aligned (just to body axis) 3D position
    %changing color to represent time - red= early, blue = late
    figure
    subplot(4,1,1:3)
    colStore = [(1/length(frames):1/length(frames):1)',zeros(length(frames),1),(1:-1/length(frames):1/length(frames))'];
    for n=1:1:length(frames)
        plot3(alignedXYZcoordsStore{n}(:,1),alignedXYZcoordsStore{n}(:,2),alignedXYZcoordsStore{n}(:,3),'color',colStore(n,:))
        hold on
    end
    plot3(0,0,0,'ko')
    xlabel('X')
    ylabel('Y')
    zlabel('Z')
    axis equal
    title('Head position over time, red = early, blue = late.  Aligned to body axis')
    subplot(4,1,4)
    plot(frames,LantTipStore-repmat(LantTipStore(1,:),size(LantTipStore,1),1))
    xlabel('Frame')
    title('3D position of left antennal tip')
    
    %raw (just to body axis) 3D position
    %changing color to represent time - red= early, blue = late
    figure
    subplot(4,1,1:3)
    colStore = [(1/length(frames):1/length(frames):1)',zeros(length(frames),1),(1:-1/length(frames):1/length(frames))'];
    for n=1:1:length(frames)
        plot3(rawXYZcoordsStore{n}(:,1),rawXYZcoordsStore{n}(:,2),rawXYZcoordsStore{n}(:,3),'color',colStore(n,:))
        hold on
    end
    plot3(0,0,0,'ko')
    xlabel('X')
    ylabel('Y')
    zlabel('Z')
    axis equal
    title('Head position over time, red = early, blue = late.  Raw data')
    subplot(4,1,4)
    title('3D position of left antennal tip')
    
    %angle about (changing) axis from axis-angle representation
    figure
    plot(frames,axisAngleStore_DEGXYZ(:,1))
    title('Angle from axis-angle representation (axis angle applies to changes each frame - just use to get idea of magnitude of rotation over time, not direction)')
    xlabel('Frame')
    
end

if plotYN
    
    %angular velocities
    figure
    plot(frames,angVel(:,1),'r')
    hold on
    plot(frames,angVel(:,2),'g')
    plot(frames,angVel(:,3),'b')
    title('Angular velocity.  Rotation about X/pitch = red, Y/roll = green, Z/yaw=blue')
    xlabel('Frame')
    

    
    %translation vector
    figure
    plot(frames,translationStore(:,1),'r')
    hold on
    plot(frames,translationStore(:,2),'g')
    plot(frames,translationStore(:,3),'b')
    title('translation vector (x=red, y=green, z=blue)')
    xlabel('Frame')   
    
    
    %euler angles
    figure
    plot(frames,sjh_rad2deg(phiRad),'r')
    hold on
    plot(frames,sjh_rad2deg(thetaRad),'g')
    plot(frames,sjh_rad2deg(psiRad),'b')
    title('Euler Angles.  Rotation about X/pitch = red, Y/roll = green, Z/yaw=blue')
    xlabel('Frame')  
    ylabel('Deg')
    
end
    
    



end %end of main function









