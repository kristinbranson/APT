%code snippet to demonstrate how to get 2D head position data from 3D data
%saved in Kine data structure.


%% getting data files
  
[f,p] = uigetfile('*.mat','Please select Kine file containing HEAD position data');
headDataFname = [p,f];
bodyDataFname = [p,f];%quick hack, some videos don't contain body data

load(headDataFname)

%specifying frames with data in them 
%frames in which have data are not specified, so taking best guess at them.
% Usual frames with data are 2900 and 3800, but this jitters around a lot so
% finding closest frames to these which actually have data. Very dirty hack, will probably not work for some instances,
% apologies
p=find(data.kine.flyhead.data.coords(1,1,:)~=0);%pages with data
[~,i]=min(abs(2900-p));%index of page with data that is closest to normal 2900 startframe
startFrame = p(i);
[~,i]=min(abs(3800-p));%index of page with data that is closest to normal 3800 endframe
endFrame = p(i);
bodyAngleFrame =1;%pivot point and body axis digitized in this frame (many videos don't have this)

%% getting video files
[f,p] = uigetfile('*.avi','Please select video file for view 1');
video1Fname = [p,f];

[f,p] = uigetfile('*.avi','Please select video file for view 2');
video2Fname = [p,f];


%% extracting head position data from kine format

%loading head position kine data 
load(headDataFname)

%getting kine data structure indices for different head points
i_LantTip =  find(ismember(data.kine.flyhead.config.points,'LantTip'));
i_RantTip =  find(ismember(data.kine.flyhead.config.points,'RantTip'));
i_LantBase =  find(ismember(data.kine.flyhead.config.points,'LantBase'));
i_RantBase =  find(ismember(data.kine.flyhead.config.points,'RantBase'));
i_ProboscisRoof =  find(ismember(data.kine.flyhead.config.points,'ProboscisRoof'));


%getting head points for start and end frame
headCoordsStart = ...
[   data.kine.flyhead.data.coords(i_LantTip,:,startFrame);...
    data.kine.flyhead.data.coords(i_RantTip,:,startFrame);...
    data.kine.flyhead.data.coords(i_LantBase,:,startFrame);...
    data.kine.flyhead.data.coords(i_RantBase,:,startFrame);...
    data.kine.flyhead.data.coords(i_ProboscisRoof,:,startFrame) ];
    
headCoordsEnd = ...
[   data.kine.flyhead.data.coords(i_LantTip,:,endFrame);...
    data.kine.flyhead.data.coords(i_RantTip,:,endFrame);...
    data.kine.flyhead.data.coords(i_LantBase,:,endFrame);...
    data.kine.flyhead.data.coords(i_RantBase,:,endFrame);...
    data.kine.flyhead.data.coords(i_ProboscisRoof,:,endFrame) ];


%% extracting body axis and head pivot data from kine format

%loading body axis and head pivot position kine data 
load(bodyDataFname)

%getting kine data structure indices for different body axis and head pivot points
i_headPivot =  find(ismember(data.kine.flyhead.config.points,'headPivot'));
i_neckCentre =  find(ismember(data.kine.flyhead.config.points,'neckCentre'));
i_thoracicAbdomenJoint =  find(ismember(data.kine.flyhead.config.points,'thoracicAbdomenJoint'));
i_LwingBase =  find(ismember(data.kine.flyhead.config.points,'LwingBase'));
i_RwingBase =  find(ismember(data.kine.flyhead.config.points,'RwingBase'));

try %only some files have body axis data
    
    %getting pivot point and body axis coordinates
    pivotPoint = data.kine.flyhead.data.coords(i_headPivot,:,bodyAngleFrame);

    bodyCoords = ...
    [   data.kine.flyhead.data.coords(i_neckCentre,:,bodyAngleFrame);...  
        data.kine.flyhead.data.coords(i_thoracicAbdomenJoint,:,bodyAngleFrame);... 
        data.kine.flyhead.data.coords(i_LwingBase,:,bodyAngleFrame);... 
        data.kine.flyhead.data.coords(i_RwingBase,:,bodyAngleFrame)];
end

%% converting back to 2D

%view 1
[ headStartView1(:,1), headStartView1(:,2) ] = dlt_3D_to_2D( data.cal.coeff.DLT_1,headCoordsStart(:,1), headCoordsStart(:,2), headCoordsStart(:,3) );
[ headEndView1(:,1), headEndView1(:,2) ] = dlt_3D_to_2D( data.cal.coeff.DLT_1,headCoordsEnd(:,1), headCoordsEnd(:,2), headCoordsEnd(:,3) );
try
    [ bodyView1(:,1), bodyView1(:,2) ] = dlt_3D_to_2D( data.cal.coeff.DLT_1,bodyCoords(:,1),bodyCoords(:,2),bodyCoords(:,3) );
end

%view 2
[headStartView2(:,1), headStartView2(:,2)  ] = dlt_3D_to_2D( data.cal.coeff.DLT_2,headCoordsStart(:,1), headCoordsStart(:,2), headCoordsStart(:,3) );
[ headEndView2(:,1), headEndView2(:,2) ] = dlt_3D_to_2D( data.cal.coeff.DLT_2,headCoordsEnd(:,1), headCoordsEnd(:,2), headCoordsEnd(:,3) );
try
    [ bodyView2(:,1), bodyView2(:,2) ] = dlt_3D_to_2D( data.cal.coeff.DLT_2,bodyCoords(:,1),bodyCoords(:,2),bodyCoords(:,3) );
end


%% plotting

figure
%view 1 start
subplot(1,2,1)
vid1_h = VideoReader(video1Fname);
imshow(read(vid1_h,startFrame),[])
hold on
plot( headStartView1(:,1), headStartView1(:,2),'ro')
hold off

%view 2 start
subplot(1,2,2)
vid2_h = VideoReader(video2Fname);
imshow(read(vid2_h,startFrame),[])
hold on
plot( headStartView2(:,1), headStartView2(:,2),'ro')
hold off

figure
%view 1 end
subplot(1,2,1)
imshow(read(vid1_h,endFrame),[])
hold on
plot( headEndView1(:,1), headEndView1(:,2),'ro')
hold off

%view 2 end
subplot(1,2,2)
imshow(read(vid2_h,endFrame),[])
hold on
plot( headEndView2(:,1), headEndView2(:,2),'ro')
hold off

try
    %body angle data (if have data in this video)
    figure
    subplot(1,2,1)
    imshow(read(vid1_h,bodyAngleFrame),[])
    hold on
    plot( bodyView1(:,1), bodyView1(:,2),'ro')
    hold off

    subplot(1,2,2)
    imshow(read(vid2_h,bodyAngleFrame),[])
    hold on
    plot( bodyView2(:,1), bodyView2(:,2),'ro')
    hold off
    
catch
        disp('No body data in this video')
end


