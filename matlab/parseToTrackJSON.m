function [movfiles,trkfiles,trxfiles,cropRois,calibrationfiles,targets,f0s,f1s] = parseToTrackJSON(jsonfile,lObj)

res = jsondecode(fileread(jsonfile));
assert(isfield(res,'toTrack'));
res = res.toTrack;
if iscell(res),
  toTrack = res{1};
  for i = 2:numel(res),
    toTrack = structappend(toTrack,res{i});
  end
elseif isstruct(res),
  toTrack = res;
end

nviews = lObj.nview;
nmovies = numel(toTrack);

assert(all(isfield(toTrack,{'movie_files','output_files'})));

needCalibration = lObj.isMultiView && ...
  ~strcmpi(lObj.trackParams.ROOT.PostProcess.reconcile3dType,'none');
if needCalibration,
  assert(isfield(toTrack,'calibration_file'));
end
needTrx = lObj.hasTrx;
if needTrx,
  assert(isfield(toTrack,'trx_files'));
end
hasCrop = isfield(toTrack,'crop_rois');
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
f0s = nan(nmovies,1);
f1s = nan(nmovies,1);

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
  if needCalibration,
    calibrationfiles{i} = toTrack(i).calibration_file;
  end
  
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
  if isfield(toTrack(i),'frame0'),
    f0s(i) = toTrack(i).frame0;
  end
  if isfield(toTrack(i),'frame1'),
    f1s(i) = toTrack(i).frame1;
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
