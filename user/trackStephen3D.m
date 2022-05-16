function trackStephen3D(lbl_file, jsonfile,varargin)

APT.setpathsmart;
nvw = 2;
pp3dtype = 'triangulate';
net_mode = DLNetMode.singleAnimal;
apt_root = APT.getRoot;

[bindpath,backend,gpu_id] = myparse(varargin, ...
  'bindpath',{'/groups','/nrs'},...
  'backend','singularity', ...
  'gpu_id',0 ...
  );

% create a dummy lObj to satisify parseToTrackJSON
lObj = struct('nview',nvw,'isMultiView',true,...
  'hasTrx',false);
lObj.trackParams.ROOT.PostProcess.reconcile3dType = 'trianglulate';

[toTrack] = parseToTrackJSON(jsonfile,lObj);
assert(~isempty(toTrack));

if size(toTrack.cropRois,2) > 1,
  cropRois = cell(size(toTrack.cropRois,1),1);
  for i = 1:size(toTrack.cropRois,1),
    cropRois{i} = cat(1,toTrack.cropRois{i,:});
  end
else
  cropRois = toTrack.cropRois;
end

for ndx = 1:size(toTrack.movfiles,1)
  crop_str = sprintf('%d %d ',toTrack.cropRois{ndx}(1,:),cropRois{ndx}(2,:));
  baseCmd = sprintf('python %s/deepnet/APT_track.py -lbl_file %s -mov %s %s -out %s %s -crop_loc %s',...
    apt_root,...
    lbl_file, toTrack.movfiles{ndx,1}, toTrack.movfiles{ndx,2},...
     toTrack.trkfiles{ndx,1}, toTrack.trkfiles{ndx,2}, ...
     crop_str);

  if strcmp(backend, 'singularity')
    cmd_str = DeepTracker.codeGenSingGeneral(baseCmd,net_mode,'bindpath',bindpath);
  else
    backend = DLBackEndClass(DLBackEnd.Docker);
    cmd_str = backend.codeGenDockerGeneral(baseCmd,'run1','bindpath',bindpath,...
      'gpuid',gpu_id,'detach',false);
  end  
  system(cmd_str);

  calibrationfile=toTrack.calibrationfiles{ndx};
  vcd = CalRig.loadCreateCalRigObjFromFile(calibrationfile);

  trkfiles = toTrack.trkfiles(ndx,:);
  [trks,tfsucc] = ...
    cellfun(@(x)DeepTracker.hlpLoadTrk(x,'rawload',true),trkfiles,'uni',0);
  tfsucc = cell2mat(tfsucc);
  if ~all(tfsucc)
    ivwFailed = find(~tfsucc);
    ivwFailedStr = num2str(ivwFailed(:)');
    error('Cannot perform 3D postprocessing; could not load trkfiles for views: %s.',ivwFailedStr);
  end

  trk1 = trks{1};
  trk2 = trks{2};                

  [trk1save,trk2save] = PostProcess.triangulate(trk1,trk2,...
    toTrack.cropRois{ndx},vcd,pp3dtype);
  save(trkfiles{1},'-append','-struct','trk1save');
  fprintf(1,'Saved/appended variables ''pTrkSingleView'', ''pTrk'', ''pTrk3d'' to trkfile %s.\n',...
        trkfiles{1});
  save(trkfiles{2},'-append','-struct','trk2save');
  fprintf(1,'Saved/appended variables ''pTrkSingleView'', ''pTrk'', to trkfile %s.\n',...
        trkfiles{2});
end

