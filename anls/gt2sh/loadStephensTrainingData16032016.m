function headVideo = loadStephensTrainingData16032016()
%
% Reformats all Stephen's digitization of head position in fly videos into
% one big struct called headVideo. 
%
% fields of headVideo:
%
% headVideo(n).frames = frame numbers (starts at 1) of video where have
% clicked on head
%
%headVideo(n).kineDataFname = original data filename - useful for stephen as it has
%number of fly and video
%
%headVideo(n).vidFname_view1/2 = filename of video corresponding data
%points are for. one field for both view 1 and 2
%
%
%headVideo(n).view1/2.LantTip = x,y coords of left antennal tip in view 1 or 2
%headVideo(n).view1/2.RantTip = x,y coords of right antennal tip in view 1 or 2
%headVideo(n).view1/2.LantBase = x,y coords of left antennal base in view 1 or 2
%headVideo(n).view1/2.RantBase = x,y coords of right antennal base in view 1 or 2
%headVideo(n).view1/2.ProboscisRoof = x,y coords of 'proboscis roof' (made up term) in view 1 or 2 

%data path
newPath = '/groups/branson/bransonlab/projects/flyHeadTracking/'; %change this to wherever you keep the data
path2replace_data = 'X:\flyHeadTrainingData16032016\'; %the part of Stephen's original data path you want to replace with newPath
path2replace_videos = 'W:\home\hustons\flp-chrimson_experiments\egFlpTrials4analysis\';%path of where videos were when digitization was done - this will be replaced by newPath too

%plot flag
plotEGframe =1; %set to 1 if want to plot example frame and data for sanity check


%file containing info on which frames have been digitized (1st frame = frame 1)
load framesClicked

%filenames of kine data
kineDataFiles = {
    'X:\flyHeadTrainingData16032016\fly178\kineData\01_fly178_kineData.mat',...
    'X:\flyHeadTrainingData16032016\fly178\kineData\02_fly178_kineData.mat',...
    'X:\flyHeadTrainingData16032016\fly178\kineData\03_fly178_kineData.mat',...
    'X:\flyHeadTrainingData16032016\fly178\kineData\04_fly178_kineData.mat',...
    'X:\flyHeadTrainingData16032016\fly178\kineData\05_fly178_kineData.mat',...
    'X:\flyHeadTrainingData16032016\fly178\kineData\06_fly178_kineData.mat',...
    'X:\flyHeadTrainingData16032016\fly178\kineData\07_fly178_kineData.mat',...
    'X:\flyHeadTrainingData16032016\fly178\kineData\08_fly178_kineData.mat',...
    'X:\flyHeadTrainingData16032016\fly178\kineData\09_fly178_kineData.mat',...
    'X:\flyHeadTrainingData16032016\fly178\kineData\10_fly178_kineData.mat',...
    'X:\flyHeadTrainingData16032016\fly212\kineData\01_fly212_kineData.mat',...
    'X:\flyHeadTrainingData16032016\fly216\kineData\01_fly216_kineData.mat',...
    'X:\flyHeadTrainingData16032016\fly219\kineData\01_fly219_kineData.mat',...
    'X:\flyHeadTrainingData16032016\fly229\kineData_300msStimuli\01_fly229_kineData.mat',...
    'X:\flyHeadTrainingData16032016\fly230\kineData\01_fly230_kineData.mat',...
    'X:\flyHeadTrainingData16032016\fly234\kineData\01_fly234_kineData.mat',...
    'X:\flyHeadTrainingData16032016\fly234\kineData\02_fly234_kineData.mat',...
    'X:\flyHeadTrainingData16032016\fly235\kineData\01_fly235_kineData.mat',...
    'X:\flyHeadTrainingData16032016\fly241\kineData_300ms_stimuli\01_fly241_kineData_300msStim.mat',...
    'X:\flyHeadTrainingData16032016\fly244\kineData\01_fly244_kineData.mat',...
    'X:\flyHeadTrainingData16032016\fly245\kineData\01_fly245_kineData.mat',...
    'X:\flyHeadTrainingData16032016\fly251\kineData\01_fly251_kineData.mat',...
    'X:\flyHeadTrainingData16032016\fly254\kineData\01_fly254_kineData.mat',...
    };


%% RUN ME 1
kineDataFiles = importdata('f:\stephenTrainingDataForMayanksTracker\kineFiles.txt');
kineDataFiles = fullfile('f:\stephenTrainingDataForMayanksTracker\pre2016_trainingData_kineFormat_blindToErrors\',kineDataFiles);

%% correcting data path based upon newpath variable
for n = 1:length(kineDataFiles) %for each kine data file
  name = kineDataFiles{n}(length(path2replace_data)+1:end);
  if ~ispc,
    name = strrep(name,'\','/');
  end
  kineDataFiles{n} = fullfile(newPath,name);
  assert(exist(kineDataFiles{n},'file')>0);
end


%% RUN ME 2

s = struct(...
  'lblCat',cell(0,1),... % int enum for type of label file
  'lblFile',cell(0,1),... % 3-level path
  'iMov',[],... % movie index within lblFile
  'movFile',[],... % [1xnview] movie fullpaths
  'movID',[],... % movie ID (standard path, movie1)
  'movID2',[],...
  'flyID',[],...
  'frm',[],... 
  'pLbl',[],... % [1xnLabelPoints*2]==[1xnphyspts*nvw*2]
  'pLblTSmin',[],... % scalar, minimum label timestamp 
  'pLblTSmax',[]); % scalar, max label timestamp 

for n = 1:length(kineDataFiles) %for each kine data file
    warnst = warning('off','MATLAB:nonExistentField');
    try
      load(kineDataFiles{n},'data');
    catch
      tmp = matfile(kineDataFiles{n});
      data = tmp.data;
    end
    warning(warnst);
    
    %saving frames that were clicked on to headVideo struct
    %headVideo(n).frames = framesClicked;
    headVideo(n).frames = zeros(0,1);
    
    %fixing path of video locations and then saving to headVideo struct
    [oldView1Path,oldView2Path] = data.images.flexload.path; %old video paths - will be wrong for current location
    [oldView1fname,oldView2fname] = data.images.flexload.file;    
    headVideo(n).vidFname_view1 = fullfile(oldView1Path,oldView1fname);
    headVideo(n).vidFname_view2 = fullfile(oldView2Path,oldView2fname); 
% % %     name = oldView1Path(length(path2replace_videos)+1:end);
% % %     if ~ispc,
% % %       name = strrep(name,'\','/');
% % %     end
% % %     headVideo(n).vidFname_view1 = fullfile(newPath,name,oldView1fname); 
% % %     assert(exist(headVideo(n).vidFname_view1,'file')>0);
% % %     name = oldView2Path(length(path2replace_videos)+1:end);
% % %     if ~ispc,
% % %       name = strrep(name,'\','/');
% % %     end
% % %     headVideo(n).vidFname_view2 = fullfile(newPath,name,oldView2fname); 
% % %     assert(exist(headVideo(n).vidFname_view2,'file')>0);
    
    %saving filename of kine data - gives Stephen the fly and trial ID
    headVideo(n).kineDataFname = kineDataFiles{n};    
    
    %intializing fields so can iteratively add data to them (don't judge me)
    headVideo(n).view1.LantTip = [];
    headVideo(n).view1.RantTip = [];
    headVideo(n).view1.LantBase = [];
    headVideo(n).view1.RantBase = [];
    headVideo(n).view1.ProboscisRoof = [];
    headVideo(n).view2.LantTip = [];
    headVideo(n).view2.RantTip = [];
    headVideo(n).view2.LantBase = [];
    headVideo(n).view2.RantBase = [];
    headVideo(n).view2.ProboscisRoof = [];
    
    tfFlyHead = isfield(data.kine,'flyhead');
    tfFlyHeadNB = isfield(data.kine,'flyhead_noBody');
    assert(xor(tfFlyHead,tfFlyHeadNB));
    if tfFlyHead
      labelmat = data.kine.flyhead.data.coords;
    else      
      labelmat = data.kine.flyhead_noBody.data.coords;
    end
    nfrm = size(labelmat,3);
    fLbled = arrayfun(@(x)nnz(labelmat(:,:,x)~=0)>0,1:nfrm);
    fLbled = find(fLbled);
        
    for fi = 1:length(fLbled) 
        frame = fLbled(fi);
        
% % %         %extracting 3D points into own variables just for my
% % %         %convinience/sanity
% % %         try %try catch to deal with fact used two different data structures
            lat_xyz = labelmat(1,:,frame); %left antenna tip 3D
            rat_xyz = labelmat(2,:,frame); %right antenna tip 3D
            lab_xyz = labelmat(3,:,frame); %left antenna base 3D
            rab_xyz = labelmat(4,:,frame); %right antenna base 3D
            pr_xyz = labelmat(5,:,frame); %proboscis roof 3D

            assert(nnz(labelmat(:,:,frame))>0);
% % %             if nnz(labelmat(:,:,frame))==0
% % %               warning('all coords are 0 for file %d (%s), frame %d, skipping',n,headVideo(n).kineDataFname,frame);
% % %               continue;
% % %             end
            
% % %         catch 
% % %             lat_xyz = data.kine.flyhead_noBody.data.coords(1,:,frame); %left antenna tip 3D
% % %             rat_xyz = data.kine.flyhead_noBody.data.coords(2,:,frame); %right antenna tip 3D
% % %             lab_xyz = data.kine.flyhead_noBody.data.coords(3,:,frame); %left antenna base 3D
% % %             rab_xyz = data.kine.flyhead_noBody.data.coords(4,:,frame); %right antenna base 3D
% % %             pr_xyz = data.kine.flyhead_noBody.data.coords(5,:,frame); %proboscis roof 3D
% % %             
% % %             if nnz(data.kine.flyhead_noBody.data.coords(:,:,frame))==0,
% % %               warning('all coords are 0 for file %d (%s), frame %d, skipping',n,headVideo(n).kineDataFname,frame);
% % %               continue;
% % %             end
% % %             
% % %         end
    
        %converting back from 3D to 2D for both views
        [ lat_xy_view1(1), lat_xy_view1(2) ] = dlt_3D_to_2D( data.cal.coeff.DLT_1, lat_xyz(1), lat_xyz(2), lat_xyz(3) ); %left antenna tip 2D view 1
        [ rat_xy_view1(1), rat_xy_view1(2) ] = dlt_3D_to_2D( data.cal.coeff.DLT_1, rat_xyz(1), rat_xyz(2), rat_xyz(3) ); %left antenna tip 2D view 1
        [ lab_xy_view1(1), lab_xy_view1(2) ] = dlt_3D_to_2D( data.cal.coeff.DLT_1, lab_xyz(1), lab_xyz(2), lab_xyz(3) ); %left antenna tip 2D view 1
        [ rab_xy_view1(1), rab_xy_view1(2) ] = dlt_3D_to_2D( data.cal.coeff.DLT_1, rab_xyz(1), rab_xyz(2), rab_xyz(3) ); %left antenna tip 2D view 1
        [ pr_xy_view1(1), pr_xy_view1(2) ] = dlt_3D_to_2D( data.cal.coeff.DLT_1, pr_xyz(1), pr_xyz(2), pr_xyz(3) ); %left antenna tip 2D view 1

        
        [ lat_xy_view2(1), lat_xy_view2(2) ] = dlt_3D_to_2D( data.cal.coeff.DLT_2, lat_xyz(1), lat_xyz(2), lat_xyz(3) ); %left antenna tip 2D view 1
        [ rat_xy_view2(1), rat_xy_view2(2) ] = dlt_3D_to_2D( data.cal.coeff.DLT_2, rat_xyz(1), rat_xyz(2), rat_xyz(3) ); %left antenna tip 2D view 1
        [ lab_xy_view2(1), lab_xy_view2(2) ] = dlt_3D_to_2D( data.cal.coeff.DLT_2, lab_xyz(1), lab_xyz(2), lab_xyz(3) ); %left antenna tip 2D view 1
        [ rab_xy_view2(1), rab_xy_view2(2) ] = dlt_3D_to_2D( data.cal.coeff.DLT_2, rab_xyz(1), rab_xyz(2), rab_xyz(3) ); %left antenna tip 2D view 1
        [ pr_xy_view2(1), pr_xy_view2(2) ] = dlt_3D_to_2D( data.cal.coeff.DLT_2, pr_xyz(1), pr_xyz(2), pr_xyz(3) ); %left antenna tip 2D view 1

        
        %storing data in xy arrays for each view
        headVideo(n).frames(end+1,1) = frame;
        headVideo(n).view1.LantTip = [headVideo(n).view1.LantTip; lat_xy_view1];
        headVideo(n).view1.RantTip = [headVideo(n).view1.RantTip; rat_xy_view1];
        headVideo(n).view1.LantBase = [headVideo(n).view1.LantBase; lab_xy_view1];
        headVideo(n).view1.RantBase = [headVideo(n).view1.RantBase; rab_xy_view1];
        headVideo(n).view1.ProboscisRoof = [headVideo(n).view1.ProboscisRoof; pr_xy_view1];
        
        headVideo(n).view2.LantTip = [headVideo(n).view2.LantTip; lat_xy_view2];
        headVideo(n).view2.RantTip = [headVideo(n).view2.RantTip; rat_xy_view2];
        headVideo(n).view2.LantBase = [headVideo(n).view2.LantBase; lab_xy_view2];
        headVideo(n).view2.RantBase = [headVideo(n).view2.RantBase; rab_xy_view2];
        headVideo(n).view2.ProboscisRoof = [headVideo(n).view2.ProboscisRoof; pr_xy_view2];
        
% % %     end

        assert(isrow(lat_xy_view1));
        
        s(end+1,1).lblCat = 'kine';
        s(end).lblFile = FSPath.nLevelFnameChar(headVideo(n).kineDataFname,3);
        s(end).iMov = 1;
        s(end).movFile = {headVideo(n).vidFname_view1 headVideo(n).vidFname_view2};
        [flyID1,movID1] = parseSHfullmovie(s(end).movFile{1});
        [flyID2,movID2] = parseSHfullmovie(s(end).movFile{2});
        assert(flyID1==flyID2);
        s(end).flyID = flyID1;
        s(end).movID = FSPath.standardPathChar(movID1);
        s(end).movID2 = FSPath.standardPathChar(movID2);
        s(end).frm = frame;
        pLbl = [...
          lat_xy_view1; rat_xy_view1; lab_xy_view1; rab_xy_view1; pr_xy_view1; ...
          lat_xy_view2; rat_xy_view2; lab_xy_view2; rab_xy_view2; pr_xy_view2];
        s(end).pLbl = pLbl(:)';
        s(end).pLblTSmin = datenum(data.save.timestamp);
        s(end).pLblTSmax = datenum(data.save.timestamp);
    end
  
    disp([num2str(n),' of ',num2str(length(kineDataFiles)),' done.'])
end
tblKine = struct2table(s);


% % % %% optional plot for sanity check
% % % 
% % % if true 
% % % 
% % %     vid2plot = 20;
% % %     frameIdx2plot = 2;
% % % 
% % %     figure
% % % 
% % %     %view 1
% % % 
% % %     subplot(1,2,1)
% % %     v1 = VideoReader(headVideo(vid2plot).vidFname_view1);
% % %     frame_v1 = read(v1,headVideo(vid2plot).frames(frameIdx2plot));
% % % 
% % %     imshow(frame_v1)
% % %     hold on
% % %     plot(headVideo(vid2plot).view1.LantTip(frameIdx2plot,1),headVideo(vid2plot).view1.LantTip(frameIdx2plot,2),'go')
% % %     plot(headVideo(vid2plot).view1.RantTip(frameIdx2plot,1),headVideo(vid2plot).view1.RantTip(frameIdx2plot,2),'ro')
% % %     plot(headVideo(vid2plot).view1.LantBase(frameIdx2plot,1),headVideo(vid2plot).view1.LantBase(frameIdx2plot,2),'gx')
% % %     plot(headVideo(vid2plot).view1.RantBase(frameIdx2plot,1),headVideo(vid2plot).view1.RantBase(frameIdx2plot,2),'rx')
% % %     plot(headVideo(vid2plot).view1.ProboscisRoof(frameIdx2plot,1),headVideo(vid2plot).view1.ProboscisRoof(frameIdx2plot,2),'co')
% % % 
% % %     %view 2
% % % 
% % %     subplot(1,2,2)
% % %     v2 = VideoReader(headVideo(vid2plot).vidFname_view2);
% % %     frame_v2 = read(v2,headVideo(vid2plot).frames(frameIdx2plot));
% % % 
% % %     imshow(frame_v2)
% % %     hold on
% % %     plot(headVideo(vid2plot).view2.LantTip(frameIdx2plot,1),headVideo(vid2plot).view2.LantTip(frameIdx2plot,2),'go')
% % %     plot(headVideo(vid2plot).view2.RantTip(frameIdx2plot,1),headVideo(vid2plot).view2.RantTip(frameIdx2plot,2),'ro')
% % %     plot(headVideo(vid2plot).view2.LantBase(frameIdx2plot,1),headVideo(vid2plot).view2.LantBase(frameIdx2plot,2),'gx')
% % %     plot(headVideo(vid2plot).view2.RantBase(frameIdx2plot,1),headVideo(vid2plot).view2.RantBase(frameIdx2plot,2),'rx')
% % %     plot(headVideo(vid2plot).view2.ProboscisRoof(frameIdx2plot,1),headVideo(vid2plot).view2.ProboscisRoof(frameIdx2plot,2),'co')
% % % 
% % %     
% % % end
