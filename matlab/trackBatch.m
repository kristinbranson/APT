% Batch Tracking 
%
% trackBatch('lObj',lObj,'jsonfile',jsonfile)
% With APT open, track multiple videos (that do not have to be part of the 
% current project) using the currently selected tracker.
% Example json files are in <APT>/examples/totrack_example*.json.
%
% trackBatch('lblfile',lblfile,'net','cpr','movfiles',movfiles,...
%  'trkfiles',trkfiles)
% Track a set of movies with the CPR tracker contained in a saved project
% file (.lbl file).
%
% trackBatch('lblfile',lblfile,'net','cpr','movfiles',movfiles,...
%  'trxfiles',trxfiles,'trkfiles',trkfiles)
% Optionally specify trxfiles for each movie to be tracked.


function trackBatch(varargin)

[lObj,jsonfile,toTrack,track_type,loargs] = ...
  myparse_nocheck(varargin,...
  'lObj',[],... % one of 'lObj' or 'lblfile' must be spec'd
  'jsonfile','',...
  'toTrack',[],...
  'track_type','track');

tfAPTOpen = ~isempty(lObj);
assert(tfAPTOpen,'Headless tracking not implemented');

if ~isempty(jsonfile),
  % read what to track from json file
  [toTrack] = parseToTrackJSON(jsonfile,lObj);
end
assert(~isempty(toTrack));

if iscell(toTrack.f0s),
  f0s = ones(size(toTrack.f0s));
  idx = ~cellfun(@isempty,toTrack.f0s);
  f0s(idx) = cell2mat(toTrack.f0s(idx));
else
  f0s = toTrack.f0s;
end
if iscell(toTrack.f1s),
  f1s = inf(size(toTrack.f1s));
  idx = ~cellfun(@isempty,toTrack.f1s);
  f1s(idx) = cell2mat(toTrack.f1s(idx));
else
  f1s = toTrack.f1s;
end
% if size(toTrack.cropRois,2) > 1,
%   cropRois = cell(size(toTrack.cropRois,1),1);
%   for i = 1:size(toTrack.cropRois,1),
%     cropRois{i} = cat(1,toTrack.cropRois{i,:});
%   end
% else
  cropRois = toTrack.cropRois;
% end
if ~iscell(toTrack.targets) && size(toTrack.movfiles,1) == 1,
  toTrack.targets = {toTrack.targets};
end
if isempty(toTrack.calibrationfiles),
  calibrationfiles = {};
elseif ischar(toTrack.calibrationfiles),
  calibrationfiles = {toTrack.calibrationfiles};
else
  calibrationfiles = toTrack.calibrationfiles;
end
assert(iscell(calibrationfiles));

totrackinfo = ToTrackInfo('movfiles',toTrack.movfiles,...
  'trxfiles',toTrack.trxfiles,'trkfiles',toTrack.trkfiles,...
  'views',1:lObj.nview,...
  'stages',1:lObj.tracker.getNumStages(),'croprois',cropRois,...
  'calibrationfiles',calibrationfiles,...
  'frm0',f0s,'frm1',f1s,...
  'trxids',toTrack.targets);
% to do: figure out how to handle linking option

% call tracker.track to do the real tracking
lObj.tracker.track('totrackinfo',totrackinfo,'track_type',track_type,'isexternal',true,loargs{:});

%   lObj.tracker.track(toTrack.movfiles,'trxfiles',toTrack.trxfiles,'trkfiles',toTrack.trkfiles,...
%     'cropRois',cropRois,'calibrationfiles',toTrack.calibrationfiles,...
%     'targets',toTrack.targets,'f0',f0s,'f1',f1s); %,'track_id',lObj.track_id);

  