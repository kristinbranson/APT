function [axisAngleDegXYZ,translations,residualErrors,scaleErrors,...
  quaternion,EulerAnglesDeg_XYZ,pivot,refHead] = ...
    APT2RT(APTfilename,flynum2bodyLUT,flynum2calibLUT,...
    predictions1orLabels0,pivot,refHead,varargin)
  
%Takes APT project containing output of Maynak's tracker and estimates
%rotation of fly's head.
%
% Inputs:
%           APTfilename = APT .lbl project containing the 5 points outputed
%           by Maynak's tracker in .labeledpos2 field.
%
%           flynum2bodyLUT = string containing filename of .csv file with 5
%           columns.  1st col = fly number, 2nd col = filename of APT .lbl
%           project containing 10 body axis points, 3rd col = video number in APT
%           project that contains body axis labels, 4th col = frame number
%           in the video that contains body axis labels, 5th column = 'A'
%           if body axis file is APT file, 'K' if a kine file.
%
%           flynum2calibLUT = filename of .csv file with 2 columns.  1st
%           col = fly number, 2nd column = location of calibration file to
%           use for this fly.
%
%           predictions1orLabels0 = set to 1 if want to analyze predictions
%           (.labeledpos2) set to 0 if want to analyze labels
%           (.labeledpos).  If 0 (labels) should always provide non-empty pivot and
%           refHead inputs also (get these by running on predictions
%           first).  If 1 can set pivot and refHead to empty and use the
%           outputs for future runs
%
%           pivot = 3D Point to assume the fly's head pivots around.  If
%           you don't have this then set to empty [] and code will estimate
%           this.  If you run this code on the same fly twice e.g. once for
%           predictions once for labels.  Run for predictions first then
%           use the pivot outputed by that run as input for labels run.
%
%           refHead = 'Standard head' to compute rotations relative to.  If
%           you don't have this then set to empty [] and code will estimate
%           this.  If you run this code on the same fly twice e.g. once for
%           predictions once for labels.  Run for predictions first then
%           use the refHead outputed by that run as input for labels run.
%
% Outputs:
%       
%           axisAngleDegXYZ = Rotations relative to lab/body reference
%           frame for each frame of each video.
%           axisAngleDegXYZ(frame#,1,video#)=magnitude of rotation in
%           Degrees. axisAngleDegXYZ(frame#,2:4,video#)=XYZ representation
%           of axis of rotation.
%
%           translations = Translation of head for each frame of each
%           video.  In mm.
%
%           residualErrors,scaleErrors = errors in rotation/translation
%           estimations
%
%           quaternion = alternate quaternion represention of rotations
%
%           pivot = 3D Point code assumed the fly's head pivots around. If
%           running this code again on same fly use this as pivot input to ensure
%           you can compare two data sets
%
%           refHead = 'Standard head' that rotations were computed relative to.  
%           If you run this code again on data from same fly use this as
%           'refHead' input to ensure two data sets are in same reference
%           frame
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
%
%
% Dependencies:
%               kine2RotationAxis.m
%               getflynum.m
%  http://www.mathworks.com/matlabcentral/fileexchange/22422-absolute-orientation
%  http://www.mathworks.com/matlabcentral/fileexchange/35475-quaternions
%  http://www.mathworks.com/matlabcentral/fileexchange/24484-geom3d
%  sjh_rad2deg (mine - easy to replicate if you don't have - just .*180/pi)
%  quat2eulzyx.m (Kine/kine_math)

[flyNum,iMov0,iMov1,iMovViewCal,gtMovs] = myparse(varargin,...
  'flyNum',[],... % optional, explicitly-supplied flyNum (instead of parsing from APTfilename)
  'iMov0',[],... % optional, start at this movie index (inclusive)
  'iMov1',[],... % optional, end at this movie index (inclusive)
  'iMovViewCal',1,... % optional, use .viewCalibrationData{iMovViewCal} for orthocam. Default value of 1 presumes all movies (gt if applicable) in APTfilename are for same fly, use same calib
  'gtMovs',false ... % if true, analyze .labeledposGT/.labeledpos2GT; iMovViewCal applies to .viewCalibrationDataGT
  );
  

%% if using labels, data is too sparse to estimate pivot and reference head
% Need to run on predictions from same fly first to get these

if (predictions1orLabels0==0) 
    if isempty(pivot)||isempty(refHead)
        error('Need to provide non-empty pivot and refHead inputs when analyzing labels.  Get these by running on predictions from the same fly first!')
    end
end


%% getting info from files

%fly Number
if isempty(flyNum)
  assert(ischar(APTfilename));
  flyNum = getflynum(APTfilename);
end
disp(['Fly ',num2str(flyNum)])

%getting calibration info

%stephen's version delete when done
fid = fopen(flynum2calibLUT);
calibTable = textscan(fid, '%d %s', 'Delimiter',',');
i = find(calibTable{1}==flyNum);
calibFname = cell2mat(calibTable{2}(i));%filename of calibration file to use for this fly

%Allen's version - put it back like this when done
% tblCalib = readtable(flynum2calibLUT);
% % fid = fopen(flynum2calibLUT);
% % calibTable = textscan(fid, '%d %s', 'Delimiter',',');
% % i = find(calibTable{1}==flyNum);
% i = find(tblCalib.fly==flyNum);
% % calibFname = cell2mat(calibTable{2}(i));%filename of calibration file to use for this fly
% calibFname = tblCalib.calibfile{i};

%extracting body axis info from flynum2bodyLUT .csv file
fid = fopen(flynum2bodyLUT);
bodyTable = textscan(fid, '%d %s %d %d %s', 'Delimiter',',');
i = find(bodyTable{1}==flyNum);
bodyFname = cell2mat(bodyTable{2}(i)); %filename of body axis project
bodyVidNum = bodyTable{3}(i); %video number within APT project that contains body axis
bodyFrame = bodyTable{4}(i); %frame number within APT project that contains body axis
bodyAPTorKine = cell2mat(bodyTable{5}(i)); % 'A' if body data is in APT project,'K' if kine project

%stimulus timing
stimulusOnOff = flyNum2stimFrames_SJH(flyNum);

%%   old stuff here for reference fix and delete


%video frame rate - usually 125
frameRate_FPS = 125;


%Number of frames before and after stimulus turns on and off to analyze
framesBeforeStimOn = 25;
framesAfterStimOff = 120;

%kine and APT indices of different points
i_LantTip =  1;
i_RantTip =  2;
i_LantBase =  3;
i_RantBase =  4;
i_ProboscisRoof =  5;
i_headPivot = 6;
i_neckCentre = 7;
i_thoracicAbdomenJoint = 8;
i_LwingBase = 9;
i_RwingBase = 10;

%% Loading and re-formatting data


%loading tracking project file for all trials for this fly
if ischar(APTfilename)
  trackedData = load(APTfilename,'-mat');
else
  trackedData = APTfilename;
end

if gtMovs
  lposFld = 'labeledposGT';
  lpos2Fld = 'labeledpos2GT';
  vcFld = 'viewCalibrationDataGT';
else
  lposFld = 'labeledpos';
  lpos2Fld = 'labeledpos2';  
  vcFld = 'viewCalibrationData';
end

%position data is either stored in normal struct or sparse struct
%depending on which version of APT used to generate APT project.
%Checking for sparse format and expanding if exists.
nmovTD = numel(trackedData.(lposFld));
assert(numel(trackedData.(lpos2Fld))==nmovTD);
for iMov = 1:nmovTD
    lpos = trackedData.(lpos2Fld){iMov};
    if isstruct(lpos)
        % new format
        trackedData.(lpos2Fld){iMov} = SparseLabelArray.full(lpos);
    else
        % old format; no action required
    end
    lpos = trackedData.(lposFld){iMov};
    if isstruct(lpos)
        % new format
        trackedData.(lposFld){iMov} = SparseLabelArray.full(lpos);
    else
        % old format; no action required
    end
end


%getting 2D->3D DLT or orthocam variables
load(strtrim(calibFname), '-regexp', '^(?!vidObj$).')
try
    dlt_side = DLT_1;
    dlt_front =DLT_2;
    DLTorOrtho = 1;%remembering that using DLT not orthocam
catch %if no variable called DLT, must be using orthocam
    orthocamObj = trackedData.(vcFld){iMovViewCal}; 
    
    % check
    orthocamObj2 = CalRig.loadCreateCalRigObjFromFile(calibFname);
    if ~isequal(orthocamObj.rvecs,orthocamObj2.rvecs)
      warningNoTrace('Orthocam object in field ''%s'' does not match object in file %s.\n',...
        vcFld,calibFname);
    end
    
    DLTorOrtho = 0;%remembering that using orthocam calibration
end

%loading body data
if bodyAPTorKine=='K' %if using kine for body
    warning off
    load(bodyFname)
    warning on
    bodyData = data;
elseif bodyAPTorKine=='A'

    bodyDataAPT = load(bodyFname,'-mat');

    %position data is either stored in normal struct or sparse struct
    %depending on which version of APT used to generate APT project.
    %Checking for sparse format and expanding if exists.
    for iMov = 1:size(bodyDataAPT.labeledpos,1)
        lpos = bodyDataAPT.labeledpos{iMov};
        if isstruct(lpos)
            % new format
            bodyDataAPT.labeledpos{iMov} = SparseLabelArray.full(lpos);
        else
            % old format; no action required
        end
    end

end


%re-formatting body data into old Kine format for convinience of using old
%functions
if bodyAPTorKine=='A' %if using APT data for body axis, reformatting it into old Kine format so don't have to alter downstream functions

    %reformatting bodyData from 2D APT format into old Kine 3D format just
    %for ease of use so don't have to alter downstream function - lazy
    totalNumPts=size(bodyDataAPT.labeledpos{bodyVidNum},1)/2;

    %extracting from .labeledPos field that contains user clicked points -
    %assuming body points are generated by user not by tracking (which is .labeledPos2)!
    headPivot_view1 = bodyDataAPT.labeledpos{bodyVidNum}(i_headPivot,:,bodyFrame);
    headPivot_view2 = bodyDataAPT.labeledpos{bodyVidNum}(totalNumPts+i_headPivot,:,bodyFrame);

    neckCentre_view1 = bodyDataAPT.labeledpos{bodyVidNum}(i_neckCentre,:,bodyFrame);
    neckCentre_view2 = bodyDataAPT.labeledpos{bodyVidNum}(totalNumPts+i_neckCentre,:,bodyFrame);

    thoracicAbdomenJoint_view1 = bodyDataAPT.labeledpos{bodyVidNum}(i_thoracicAbdomenJoint,:,bodyFrame);
    thoracicAbdomenJoint_view2 = bodyDataAPT.labeledpos{bodyVidNum}(totalNumPts+i_thoracicAbdomenJoint,:,bodyFrame);

    LwingBase_view1 = bodyDataAPT.labeledpos{bodyVidNum}(i_LwingBase,:,bodyFrame);
    LwingBase_view2 = bodyDataAPT.labeledpos{bodyVidNum}(totalNumPts+i_LwingBase,:,bodyFrame);

    RwingBase_view1 = bodyDataAPT.labeledpos{bodyVidNum}(i_RwingBase,:,bodyFrame);
    RwingBase_view2 = bodyDataAPT.labeledpos{bodyVidNum}(totalNumPts+i_RwingBase,:,bodyFrame);



    if DLTorOrtho==1 %if using DLT for 2D->3D

        tempReconfu = reconfu([DLT_1,DLT_2], [headPivot_view1(1),headPivot_view1(2),headPivot_view2(1),headPivot_view2(2)]);
        bodyData.kine.flyhead.data.coords(i_headPivot,:,bodyFrame) = tempReconfu(1:3);

        tempReconfu = reconfu([DLT_1,DLT_2], [neckCentre_view1(1),neckCentre_view1(2),neckCentre_view2(1),neckCentre_view2(2)]);
        bodyData.kine.flyhead.data.coords(i_neckCentre,:,bodyFrame) =  tempReconfu(1:3);

        tempReconfu = reconfu([DLT_1,DLT_2], [thoracicAbdomenJoint_view1(1),thoracicAbdomenJoint_view1(2),thoracicAbdomenJoint_view2(1),thoracicAbdomenJoint_view2(2)]);
        bodyData.kine.flyhead.data.coords(i_thoracicAbdomenJoint,:,bodyFrame)=  tempReconfu(1:3);

        tempReconfu = reconfu([DLT_1,DLT_2], [LwingBase_view1(1),LwingBase_view1(2),LwingBase_view2(1),LwingBase_view2(2)]); 
        bodyData.kine.flyhead.data.coords(i_LwingBase,:,bodyFrame)= tempReconfu(1:3);

        tempReconfu = reconfu([DLT_1,DLT_2], [RwingBase_view1(1),RwingBase_view1(2),RwingBase_view2(1),RwingBase_view2(2)]);    
        bodyData.kine.flyhead.data.coords(i_RwingBase,:,bodyFrame)= tempReconfu(1:3);

    else %if using orthocam calibration for 2D->3D

        bodyData.kine.flyhead.data.coords(i_headPivot,:,bodyFrame) = stereoTriangulate(orthocamObj,headPivot_view1',headPivot_view2');

        bodyData.kine.flyhead.data.coords(i_neckCentre,:,bodyFrame) =  stereoTriangulate(orthocamObj,neckCentre_view1',neckCentre_view2');

        bodyData.kine.flyhead.data.coords(i_thoracicAbdomenJoint,:,bodyFrame)=  stereoTriangulate(orthocamObj,thoracicAbdomenJoint_view1',thoracicAbdomenJoint_view2');

        bodyData.kine.flyhead.data.coords(i_LwingBase,:,bodyFrame)= stereoTriangulate(orthocamObj,LwingBase_view1',LwingBase_view2');

        bodyData.kine.flyhead.data.coords(i_RwingBase,:,bodyFrame)= stereoTriangulate(orthocamObj,RwingBase_view1',RwingBase_view2');

    end

elseif bodyAPTorKine=='K' 
    %do nothing
else
    error('ERR:missingBody','Non recognized bodyAPTorKine varaible - should be A for APT or K for kine!')
end


%% turning 2D tracked data into 3D positions



if nmovTD==0
    warning('no tracking data in project')
end

if isempty(iMov0)
  iMov0 = 1;
end
if isempty(iMov1)
  iMov1 = nmovTD;
end
clear threeD_pos
tic
disp('Converting from 2D->3D.  Very slow...')
for vidIdx=iMov0:iMov1 %for each video


    totalNumPts = size(trackedData.(lpos2Fld){vidIdx},1)/2;


    %loading two views of same trial - ASSUMING THEY ARE SEQUENTIAL ENTRIES
    %IN TRACKING PROJECT!!
    if predictions1orLabels0 ==1
        view1_2D_data = trackedData.(lpos2Fld){vidIdx}(1:5,:,:);
        view2_2D_data = trackedData.(lpos2Fld){vidIdx}(1+totalNumPts:5+totalNumPts,:,:);
    elseif predictions1orLabels0 ==0
        view1_2D_data = trackedData.(lposFld){vidIdx}(1:5,:,:);
        view2_2D_data = trackedData.(lposFld){vidIdx}(1+totalNumPts:5+totalNumPts,:,:);
    else
        error('I can''t tell if you want to analyze predictions or labels, check inputs')
    end


    
    
    %doing 2D->3D for each point over all time points
    % threeD_pos = 3d positions where p in threeD_pos{t}{p} gives data for
    % body part/point p in trial t,  rows are video frames and columns are xyz.
        if DLTorOrtho==1 %if using DLT for 2D->3D
            for p=1:size(view1_2D_data,1)%for each of head points
                for fr =1:size(view1_2D_data,3) %for each frame

                    if ~isnan(squeeze(view1_2D_data(p,1,fr)))
                        tempxyzblah = reconfu([DLT_1,DLT_2],[squeeze(view1_2D_data(p,1,fr)),squeeze(view1_2D_data(p,2,fr)),squeeze(view2_2D_data(p,1,fr)),squeeze(view2_2D_data(p,2,fr))]);
                        P(:,p,fr) = tempxyzblah(1:3);
                        threeD_pos{vidIdx}{p}(fr,1:3) = P(:,p,fr);
                    else
                        threeD_pos{vidIdx}{p}(fr,1:3) = nan(1,3);
                    end

                end
            end
        else % if using orthocam calibration 
            for p=1:size(view1_2D_data,1)%for each of head points
                 valid_pts = ~isnan(view1_2D_data(1,1,:));
                 temp3D = permute( stereoTriangulate(orthocamObj,...
                   permute(view1_2D_data(p,1:2,valid_pts),[2,3,1]),...
                   permute(view2_2D_data(p,1:2,valid_pts),[2,3,1])), [2,1]);
                 init_pts = nan(size(view1_2D_data,3),3);
                 init_pts(valid_pts,:) = temp3D;
                 threeD_pos{vidIdx}{p}(:,1:3) = init_pts;
%                     for fr =1:size(view1_2D_data,3) %for each frame
%                         if ~isnan(squeeze(view1_2D_data(p,1,fr)))
%                             threeD_pos{vidIdx}{p}(fr,1:3) = permute( stereoTriangulate(orthocamObj,squeeze(view1_2D_data(p,1:2,fr))',squeeze(view2_2D_data(p,1:2,fr))'), [2,1]);            
%                         else
%                             threeD_pos{vidIdx}{p}(fr,1:3) = nan(1,3);
%                         end
%                     end                
            end
        end


end
disp('..done')
toc


%% If no pivot point provided using all data to estimate best pivot point to describe all head positions -slow!

if isempty(pivot)
  assert(iMov0==1 && iMov1==size(threeD_pos,2));  
  
    pivotProvided = 0;%1 for pivot was provided as an input, 0 when not
    
    %just getting all data for all videos and frames in same array
    allHeadCoords=[];

    % points = n x 3 x t 3D array.  Each row is a diffrent 3D point, each
    %       of 3 columns are the xyz coords of the point, each 3rd dimension is
    %       a different time point where the 3D points are rotated relative to
    %       other time points about a pivot.  origin of xyz points determines
    %       the starting point in the optimization to find the pivot.


    counter = 0;
    for trial = 1:size(threeD_pos,2)
        for point = 1:size(threeD_pos{1},2)
            for t = 1:size(threeD_pos{1}{1},1)
                if ~isnan(threeD_pos{trial}{point}(t,1))
                    counter = counter+1;
                    allHeadCoords(point,1:3,counter) = threeD_pos{trial}{point}(t,1:3);
                end
            end
        end
    end


    allHeadCoords(:,:, 1) = [];

    %removing empty frames - shouldn't be any for predictions but will exist
    %for labels (untested)
    i=isnan(allHeadCoords(1,1,:));
    allHeadCoords(:,:,i)=[];

    %just gettting pivot point user clicked on
    %i_headPivot =  find(ismember(bodyData.kine.flyhead.config.points,'headPivot'));
    userPivotPoint = bodyData.kine.flyhead.data.coords(i_headPivot,:,bodyFrame);


    %estimating actual pivot point that best describes all head rotations
    %downsampling if have too many points
    clear dsmpld_allHeadCoords
    maxt= 100;
    if size(allHeadCoords,3)>maxt
        downfac = round(size(allHeadCoords,3)/maxt);
        for p = 1:size(allHeadCoords,1)
            for xyz=1:3
                dsmpld_allHeadCoords(p,xyz,:) = downsample(allHeadCoords(p,xyz,:),downfac);
            end
        end
    else
        dsmpld_allHeadCoords = allHeadCoords;
    end

    disp('Estimating head pivot point. Slow...')
    [pivot]= estimatePivot(dsmpld_allHeadCoords,userPivotPoint);
    disp('...done.')
    
else
    
    
    pivotProvided = 1;%1 for pivot was provided as an input, 0 when not

end

%% if no reference head provided using all non-stimulus data to estimate best reference point for head rotations


if isempty(refHead)
    assert(iMov0==1 && iMov1==size(threeD_pos,2));  

    refHeadProvided = 0;%1 for refHead was provided as an input, 0 when not
    
    %just getting all data for all files and frames in same array
    allHeadCoords=[];
    %putting all head data from all expierments, all frames into one array
    %that can use to estimate pivot point from
    %
    % points = n x 3 x t 3D array.  Each row is a diffrent 3D point, each
    %       of 3 columns are the xyz coords of the point, each 3rd dimension is
    %       a different time point where the 3D points are rotated relative to
    %       other time points about a pivot.  origin of xyz points determines
    %       the starting point in the optimization to find the pivot.

    counter = 0;
    for trial = 1:size(threeD_pos,2)
        for sp = 1:length(stimulusOnOff)-1%for each stimulus period except last one (usually 6 stimuli per videos)
            % for non-stimulus period use roughly 1000 ms after stimulus started (700ms after stimulus ended) -
            % stimulus will restart at 1300 ms after stimulus orginially started.  So
            % 1000 ms 1s shift i.e.
            fakeStimShiftFrames = round(1*frameRate_FPS);
            st = stimulusOnOff(sp,1) - framesBeforeStimOn;%real start and end used
            en = stimulusOnOff(sp,2) + framesAfterStimOff;
            st_c = st + fakeStimShiftFrames;
            en_c = en + fakeStimShiftFrames;
            for t = st_c:en_c

                if ~isnan(threeD_pos{trial}{point}(t,1))
                    counter = counter+1;
                    for point = 1:size(threeD_pos{trial},2)
                        allHeadCoords(point,1:3,counter) = threeD_pos{trial}{point}(t,1:3) ;
                    end
                else
                    warning('NaNs in data!  Are you using labels?  If so, run on predictions first to get pivot and refHead inputs.')
                end

            end
        end

    end



    for point = 1:size(threeD_pos{1},2)
        refHead(point,:) = [median(allHeadCoords(point,1,:)),median(allHeadCoords(point,2,:)),median(allHeadCoords(point,3,:))];
    end



else
        refHeadProvided = 1;%1 for refHead was provided as an input, 0 when not

end


%% running APTtracking2RotationSequence over all data

counter = 0;
cntrlCounter=0;

max_vid_size = max(cellfun(@(x) size(x{1},1), threeD_pos(iMov0:iMov1))); % max num frms
nvids = iMov1-iMov0+1;
%nvids = size(threeD_pos,2);
translations = nan(max_vid_size, 3, nvids);
axisAngleDegXYZ = nan(max_vid_size, 4, nvids);
quaternion = nan(max_vid_size, 4, nvids);
frameStore = nan(max_vid_size, 1, nvids);
residualErrors = nan(max_vid_size, 1, nvids);
scaleErrors = nan(max_vid_size, 1, nvids);
rawXYZcoordsStore = cell(1, max_vid_size, nvids);
alignedXYZcoordsStore = cell(1,max_vid_size, nvids);
EulerAnglesDeg_XYZ = nan(max_vid_size, 3, nvids);

for vid = iMov0:iMov1 %1:size(threeD_pos,2)%for each video in experiment

    counter = counter +1;

    clear headData
    headData.kine.flyhead.data.coords(i_LantTip,1:3,:) = threeD_pos{vid}{i_LantTip}(:,1:3)';
    headData.kine.flyhead.data.coords(i_RantTip,1:3,:) = threeD_pos{vid}{i_RantTip}(:,1:3)';
    headData.kine.flyhead.data.coords(i_LantBase,1:3,:) = threeD_pos{vid}{i_LantBase}(:,1:3)';
    headData.kine.flyhead.data.coords(i_RantBase,1:3,:) = threeD_pos{vid}{i_RantBase}(:,1:3)';
    headData.kine.flyhead.data.coords(i_ProboscisRoof,1:3,:) = threeD_pos{vid}{i_ProboscisRoof}(:,1:3)';

    vid_size = size(threeD_pos{vid}{1},1);
    
    [t, aa, q, fs,...
    re, se, refHeadRet, rawXYZ, ...
    alignedXYZ,Eul]=threeD2RT(headData,bodyData,pivot,bodyFrame,frameRate_FPS, 0, refHead);

    translations(1:vid_size,:,counter)=t;
    axisAngleDegXYZ(1:vid_size,:,counter)=aa(:,4:7);
    quaternion(1:vid_size,:,counter)=q;
    frameStore(1:vid_size,:,counter)=fs;
    residualErrors(1:vid_size,:,counter)=re;
    scaleErrors(1:vid_size,:,counter)=se;
    refHeadReturned = refHeadRet;
    rawXYZcoordsStore(:,1:vid_size,counter)=rawXYZ;
    alignedXYZcoordsStore(:,1:vid_size,counter)=alignedXYZ;
    EulerAnglesDeg_XYZ(1:vid_size,:,counter)=Eul;


end



