function toTrackOut = parseToTrackJSON(jsonfile,lObj)

jsonData = jsondecode(fileread(jsonfile));
assert(isfield(jsonData,'toTrack'));

res = jsonData.toTrack;
if iscell(res),
  toTrack = res{1};
  for i = 2:numel(res),
    toTrack = structappend(toTrack,res{i});
  end
elseif isstruct(res),
  toTrack = res;
end

nviews = lObj.nview;
isma = lObj.maIsMA;
nmovies = numel(toTrack);

assert(all(isfield(toTrack,{'movie_files','output_files'})),'movie_files and output_files must be specified');

needCalibration = lObj.isMultiView && ...
  ~strcmpi(lObj.trackParams.ROOT.PostProcess.reconcile3dType,'none');
if needCalibration,
  assert(isfield(toTrack,'calibration_file'),'calibration_file must be specified');
end
needTrx = lObj.hasTrx;
if needTrx,
  assert(isfield(toTrack,'trx_files'),'trx_files must be specified');
end
hasCrop = isfield(toTrack,'crop_rois');
% if isma
%   assert(isfield(toTrack,'detect_files'),'Detect files must be specified');
% end

% 
% movfiles:  nmovies x nviews
% % cropRois: if tfexternal, cell array of [nviewx4]
% [trxfiles,trkfiles,f0,f1,cropRois,targets,iview] = myparse(varargin,...
%   'trxfiles',{},'trkfiles',{},'f0',[],'f1',[],'cropRois',{},'targets',{},...
%   'iview',nan... % used only if tfSerialMultiMov; CURRENTLY UNSUPPORTED
%   );

movfiles = cell(nmovies,nviews);
trxfiles = cell(nmovies,nviews);
trkfiles = cell(nmovies,nviews);
cropRois = cell(nmovies,1);
calibrationfiles = cell(nmovies,1);
targets = cell(nmovies,1);
f0s = cell(nmovies,1);
f1s = cell(nmovies,1);
detectfiles = cell(nmovies,nviews);

for i = 1:nmovies,
  
  % input movie locations
  movfiles(i,:) = parseViews(toTrack(i).movie_files,nviews,true);

  % output trk files
  trkfiles(i,:) = parseViews(toTrack(i).output_files,nviews,true);
  
  % trx files - multitarget only
  if needTrx,
    trxfiles(i,:) = parseViews(toTrack(i).trx_files,nviews,true);
  end
  
  % calibration file - multiview only
  %if needCalibration,
  if isfield(toTrack,'calibration_file')
    calibrationfiles{i} = toTrack(i).calibration_file;
  end
  %end
  
  if hasCrop && ~isempty(toTrack(i).crop_rois),
    cropRois_curr = parseViews(toTrack(i).crop_rois,nviews,false);
    cropRois{i} = nan(nviews,4);
    for ivw = 1:nviews,
      if isempty(cropRois_curr{ivw}),
        continue;
      end
      cropRois{i}(ivw,:) = cropRois_curr{ivw}(:)';
    end
  end
  
  % targets to track
  if isfield(toTrack(i),'targets') && lObj.hasTrx,
    targets{i} = toTrack(i).targets;
  end
  
  % frames to track
  if isfield(toTrack(i),'frame0') && ~isempty(toTrack(i).frame0),
    f0s{i} = toTrack(i).frame0;
  end
  if isfield(toTrack(i),'frame1') && ~isempty(toTrack(i).frame1),
    f1s{i} = toTrack(i).frame1;
  end

  % if isma
  %   detectfiles(i,:) = parseViews(toTrack(i).detect_files,nviews,true);
  % end
  
  % AL: looks like should be moved outside loop
  toTrackOut = struct;
  toTrackOut.movfiles = movfiles;
  toTrackOut.trkfiles = trkfiles;
  toTrackOut.trxfiles = trxfiles;
  toTrackOut.cropRois = cropRois;
  toTrackOut.calibrationfiles = calibrationfiles;
  toTrackOut.targets = targets;
  toTrackOut.f0s = f0s;
  toTrackOut.f1s = f1s;
  % toTrackOut.detectfiles = detectfiles;

  % Add linking configuration fields only if they were present in JSON
  if isfield(jsonData, 'link_type')
    toTrackOut.link_type = jsonData.link_type;
  end
  if isfield(jsonData, 'id_maintain_identity')
    toTrackOut.id_maintain_identity = jsonData.id_maintain_identity;
  end
  
end

function sviews = parseViews(s,nviews,required)

if nargin < 3,
  required = true;
end

assert(~isempty(s));
sviews = cell(1,nviews);

if isstruct(s),
  fns = fieldnames(s);
  res = regexp(fns,'^([Vv]iew)?[\s_]?(\d+)$','once','tokens');
  vwdecode = nan(1,numel(fns));
  for i = 1:numel(fns),
    if isempty(res{i}),
      continue;
    end
    vwdecode(i) = str2double(res{i}{end});
  end
  if any(vwdecode==0),
    vwdecode = vwdecode + 1;
  end
  [ism,vwidx] = ismember(1:nviews,vwdecode);
  if required,
    assert(all(ism));
  end
  
  for ivw = 1:nviews,
    if ~ism(ivw),
      continue;
    end
    sviews{ivw} = s.(fns{vwidx(ivw)});
  end
elseif iscell(s),
  ndecode = min(numel(s),nviews);
  sviews(1:ndecode) = s(1:ndecode);
else
  sviews{1} = s;
end
