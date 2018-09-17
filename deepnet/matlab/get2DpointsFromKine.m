function res = get2DpointsFromKine(varargin)

res = struct;

[headDataFname,video1Fname,video2Fname,DEBUG,debugfig] = ...
  myparse(varargin,'headDataMatFile','',...
  'view1VideoFile','','view2VideoFile','',...
  'debug',0,...
  'debugfig',[]);

npts = 5;

%code snippet to demonstrate how to get 2D head position data from 3D data
%saved in Kine data structure.


%% getting data files

if isempty(headDataFname),
  [f,p] = uigetfile('*.mat','Please select Kine file containing HEAD position data');
  if ~ischar(f),
    return;
  end
  headDataFname = [p,f];
else
  if ~exist(headDataFname,'file'),
    error('Kine file %s does not exist',headDataFname);
  end
end
defaultDir = fileparts(headDataFname);
bodyDataFname = headDataFname;%quick hack, some videos don't contain body data
res.headDataMatFile = headDataFname;
res.bodyDataMatFile = bodyDataFname;

try
  load(headDataFname,'data');
catch 
  tmp = matfile(headDataFname);
  data = tmp.data;  
end

%getting kine data structure indices for different head points
i_LantTip =  find(ismember(data.kine.flyhead.config.points,'LantTip'),1);
i_RantTip =  find(ismember(data.kine.flyhead.config.points,'RantTip'),1);
i_LantBase =  find(ismember(data.kine.flyhead.config.points,'LantBase'),1);
i_RantBase =  find(ismember(data.kine.flyhead.config.points,'RantBase'),1);
i_ProboscisRoof =  find(ismember(data.kine.flyhead.config.points,'ProboscisRoof'),1);
i_Head = [i_LantTip,i_RantTip,i_LantBase,i_RantBase,i_ProboscisRoof];

% find frames which have data
frameidx=find(sum(data.kine.flyhead.data.coords(i_Head,1,:)~=0,1)>=npts);%pages with data
nlabeled = numel(frameidx);
if nlabeled == 0,
  fprintf('No frames labeled in %s.\n',headDataFname);
  return;
end
res.frameidx = frameidx;

%specifying frames with data in them
%frames in which have data are not specified, so taking best guess at them.
% Usual frames with data are 2900 and 3800, but this jitters around a lot so
% finding closest frames to these which actually have data. Very dirty hack, will probably not work for some instances,
% apologies
[~,i]=min(abs(2900-frameidx));%index of page with data that is closest to normal 2900 startframe
startFrame = frameidx(i);
[~,i]=min(abs(3800-frameidx));%index of page with data that is closest to normal 3800 endframe
endFrame = frameidx(i);
bodyAngleFrame =1;%pivot point and body axis digitized in this frame (many videos don't have this)
res.startFrame = startFrame;
res.endFrame = endFrame;
res.bodyAngleFrame = bodyAngleFrame;

%% extracting head position data from kine format

headCoords = ...
  [   data.kine.flyhead.data.coords(i_LantTip,:,frameidx);...
  data.kine.flyhead.data.coords(i_RantTip,:,frameidx);...
  data.kine.flyhead.data.coords(i_LantBase,:,frameidx);...
  data.kine.flyhead.data.coords(i_RantBase,:,frameidx);...
  data.kine.flyhead.data.coords(i_ProboscisRoof,:,frameidx) ];

res.headCoords = headCoords;

%
% %getting head points for start and end frame
% headCoordsStart = ...
% [   data.kine.flyhead.data.coords(i_LantTip,:,startFrame);...
%     data.kine.flyhead.data.coords(i_RantTip,:,startFrame);...
%     data.kine.flyhead.data.coords(i_LantBase,:,startFrame);...
%     data.kine.flyhead.data.coords(i_RantBase,:,startFrame);...
%     data.kine.flyhead.data.coords(i_ProboscisRoof,:,startFrame) ];
%
% headCoordsEnd = ...
% [   data.kine.flyhead.data.coords(i_LantTip,:,endFrame);...
%     data.kine.flyhead.data.coords(i_RantTip,:,endFrame);...
%     data.kine.flyhead.data.coords(i_LantBase,:,endFrame);...
%     data.kine.flyhead.data.coords(i_RantBase,:,endFrame);...
%     data.kine.flyhead.data.coords(i_ProboscisRoof,:,endFrame) ];


%% extracting body axis and head pivot data from kine format

%loading body axis and head pivot position kine data
if ~strcmp(bodyDataFname,headDataFname),
  load(bodyDataFname,'data');
end

%getting kine data structure indices for different body axis and head pivot points
i_headPivot =  find(ismember(data.kine.flyhead.config.points,'headPivot'),1);
i_neckCentre =  find(ismember(data.kine.flyhead.config.points,'neckCentre'),1);
i_thoracicAbdomenJoint =  find(ismember(data.kine.flyhead.config.points,'thoracicAbdomenJoint'),1);
i_LwingBase =  find(ismember(data.kine.flyhead.config.points,'LwingBase'),1);
i_RwingBase =  find(ismember(data.kine.flyhead.config.points,'RwingBase'),1);

try %#ok<TRYNC> %only some files have body axis data
  
  %getting pivot point and body axis coordinates
  pivotPoint = data.kine.flyhead.data.coords(i_headPivot,:,bodyAngleFrame);
  res.pivotPoint = pivotPoint;
  
  bodyCoords = ...
    [   data.kine.flyhead.data.coords(i_neckCentre,:,bodyAngleFrame);...
    data.kine.flyhead.data.coords(i_thoracicAbdomenJoint,:,bodyAngleFrame);...
    data.kine.flyhead.data.coords(i_LwingBase,:,bodyAngleFrame);...
    data.kine.flyhead.data.coords(i_RwingBase,:,bodyAngleFrame)];
  res.bodyCoords = bodyCoords;
  
end

%% converting back to 2D

%view 1
npts = size(headCoords,1);
headView1 = nan([npts,2,nlabeled]);
headView2 = nan([npts,2,nlabeled]);
for i = 1:nlabeled,
  [ headView1(:,1,i), headView1(:,2,i) ] = dlt_3D_to_2D( data.cal.coeff.DLT_1,headCoords(:,1,i), headCoords(:,2,i), headCoords(:,3,i) );
  [ headView2(:,1,i), headView2(:,2,i) ] = dlt_3D_to_2D( data.cal.coeff.DLT_2,headCoords(:,1,i), headCoords(:,2,i), headCoords(:,3,i) );
end
res.headView1 = headView1;
res.headView2 = headView2;

% [ headStartView1(:,1), headStartView1(:,2) ] = dlt_3D_to_2D( data.cal.coeff.DLT_1,headCoordsStart(:,1), headCoordsStart(:,2), headCoordsStart(:,3) );
% [ headEndView1(:,1), headEndView1(:,2) ] = dlt_3D_to_2D( data.cal.coeff.DLT_1,headCoordsEnd(:,1), headCoordsEnd(:,2), headCoordsEnd(:,3) );
try %#ok<TRYNC>
  [ bodyView1(:,1), bodyView1(:,2) ] = dlt_3D_to_2D( data.cal.coeff.DLT_1,bodyCoords(:,1),bodyCoords(:,2),bodyCoords(:,3) );
  res.bodyView1 = bodyView1;
end

%view 2
% [headStartView2(:,1), headStartView2(:,2)  ] = dlt_3D_to_2D( data.cal.coeff.DLT_2,headCoordsStart(:,1), headCoordsStart(:,2), headCoordsStart(:,3) );
% [ headEndView2(:,1), headEndView2(:,2) ] = dlt_3D_to_2D( data.cal.coeff.DLT_2,headCoordsEnd(:,1), headCoordsEnd(:,2), headCoordsEnd(:,3) );
try %#ok<TRYNC>
  [ bodyView2(:,1), bodyView2(:,2) ] = dlt_3D_to_2D( data.cal.coeff.DLT_2,bodyCoords(:,1),bodyCoords(:,2),bodyCoords(:,3) );
  res.bodyView2 = bodyView2;
end


%% plotting


if DEBUG > 0,
  
  % getting video files

  if isempty(video1Fname),
    [f,p] = uigetfile('*.avi','Please select video file for view 1',defaultDir);
    if ~ischar(f),
      return;
    end
    video1Fname = [p,f];
  end
  
  if isempty(video2Fname),
    [f,p] = uigetfile('*.avi','Please select video file for view 2',defaultDir);
    if ~ischar(f),
      return;
    end
    video2Fname = [p,f];
  end
  
  res.view1VideoFile = video1Fname;
  res.view2VideoFile = video2Fname;  
  
  colors = jet(npts);
  if isempty(debugfig),
    debugfig = figure;
  else
    figure(debugfig);
    clf;
  end
  
  hax = createsubplots(2,nlabeled,.01);
  hax = reshape(hax,[2,nlabeled]);
  
  fprintf('Opening video file %s...\n',video1Fname);
  vid1_h = VideoReader(video1Fname);
  fprintf('Opening video file %s...\n',video2Fname);
  vid2_h = VideoReader(video2Fname);
  
  for i = 1:nlabeled,
    image(read(vid1_h,frameidx(i)),'Parent',hax(1,i));
    axis(hax(1,i),'image','off');
    hold(hax(1,i),'on');
    for j = 1:npts,
      plot(hax(1,i),headView1(j,1,i),headView1(j,2,i),'o','Color',colors(j,:));
    end
    text(5,5,num2str(frameidx(i)),'Color','w','HorizontalAlignment','left','VerticalAlignment','top','Parent',hax(1,i));
    
    image(read(vid2_h,frameidx(i)),'Parent',hax(2,i));
    axis(hax(2,i),'image','off');
    hold(hax(2,i),'on');
    for j = 1:npts,
      plot(hax(2,i),headView2(j,1,i),headView2(j,2,i),'o','Color',colors(j,:));
    end
    
  end
%   
%   try
%     %body angle data (if have data in this video)
%     figure
%     subplot(1,2,1)
%     imshow(read(vid1_h,bodyAngleFrame),[])
%     hold on
%     plot( bodyView1(:,1), bodyView1(:,2),'ro')
%     hold off
%     
%     subplot(1,2,2)
%     imshow(read(vid2_h,bodyAngleFrame),[])
%     hold on
%     plot( bodyView2(:,1), bodyView2(:,2),'ro')
%     hold off
%     
%   catch %#ok<CTCH>
%     disp('No body data in this video')
%   end
  res.debugfig = debugfig;

end
