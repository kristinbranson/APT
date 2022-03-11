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

[lObj,jsonfile,toTrack,lblfile,net,movfiles,trxfiles,trkfiles,loargs] = ...
  myparse_nocheck(varargin,...
  'lObj',[],... % one of 'lObj' or 'lblfile' must be spec'd
  'jsonfile','',...
  'toTrack',[],...
  'lblfile',[],... % args below only apply when 'lblfile' is spec'd
  'net',[],...  % 'cpr' or char nettype
  'movfiles',[],... % [nmov] cellstr 
  'trxfiles',[],... % optional; [nmov] cellstr 
  'trkfiles',[] ... % [nmov] cellstr
  );

tfAPTOpen = ~isempty(lObj);
tfLbl = ~isempty(lblfile);
if ~xor(tfAPTOpen,tfLbl)
  error('Exactly one of ''lObj'' or ''lblfile'' must be specified.');
end

if tfAPTOpen
  if ~isempty(jsonfile),
    % read what to track from json file
    [toTrack] = parseToTrackJSON(jsonfile,lObj);
  end
  assert(~isempty(toTrack));

  if iscell(toTrack.f0s),
    f0s = cell2mat(toTrack.f0s);
  else
    f0s = toTrack.f0s;
  end
  if iscell(toTrack.f1s),
    f1s = cell2mat(toTrack.f1s);
  else
    f1s = toTrack.f1s;
  end
  if size(toTrack.cropRois,2) > 1,
    cropRois = cell(size(toTrack.cropRois,1),1);
    for i = 1:size(toTrack.cropRois,1),
      cropRois{i} = cat(1,toTrack.cropRois{i,:});
    end
  else
    cropRois = toTrack.cropRois;
  end
  if ~iscell(toTrack.targets) && size(toTrack.movfiles,1) == 1,
    toTrack.targets = {toTrack.targets};
  end


  % call tracker.track to do the real tracking
  lObj.tracker.track(toTrack.movfiles,'trxfiles',toTrack.trxfiles,'trkfiles',toTrack.trkfiles,...
    'cropRois',cropRois,'calibrationfiles',toTrack.calibrationfiles,...
    'targets',toTrack.targets,'f0',f0s,'f1',f1s,'track_id',lObj.track_id);
else
  if ~strcmp(net,'cpr')
    error('Currently only supported for ''net''==''cpr''.');
  end
  nmov = numel(movfiles);
  if ~(iscellstr(movfiles) || isstring(movfiles))
    error('''movfiles'' must be a string array or cell array of strings.');
  end
  if ~isempty(trxfiles)
    if numel(trxfiles)~=nmov
      error('''movfiles'' and ''trxfiles'' must have the same number of elements.');
    end
    if ~(iscellstr(trxfiles) || isstring(trxfiles))
      error('''trxfiles'' must be a string array or cell array of strings.');
    end
  end
  if numel(trkfiles)~=nmov
    error('''movfiles'' and ''trkfiles'' must have the same number of elements.');
  end
  if ~(iscellstr(trkfiles) || isstring(trkfiles))
    error('''trkfiles'' must be a string array or cell array of strings.');
  end
  
  APTCluster(lblfile,'track2',movfiles,trxfiles,trkfiles,loargs{:});
end
  
  