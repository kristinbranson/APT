% trackBatch('lObj',lObj,'jsonfile',jsonfile)
% Track multiple videos that do not have to be part of the current APT
% project. 
% This is only the start of this function. Currently, only reading details
% of what to track from a json file is implemented. 
% Example json files are in 
% examples/totrack_example*.json
function trackBatch(varargin)

[lObj,jsonfile] = myparse(varargin,'lObj',[],'jsonfile','');

assert(~isempty(lObj));
assert(~isempty(jsonfile));

% read what to track from json file
[movfiles,trkfiles,trxfiles,cropRois,calibrationfiles,targets,f0s,f1s] = ...
  parseToTrackJSON(jsonfile,lObj);

% currently dumb -- f0 and f1 must be scalars that apply to all movies
% TODO: fix this
if isempty(f0s),
  f0 = [];
else
  if isnan(f0s(1)),
    f0 = [];
  else
    f0 = f0s(1);
  end
end
if isempty(f1s),
  f1 = [];
else
  if isnan(f1s(1)),
    f1 = [];
  else
    f1 = f1s(1);
  end
end    

% call tracker.track to do the real tracking
lObj.tracker.track(movfiles,'trxfiles',trxfiles,'trkfiles',trkfiles,...
  'cropRois',cropRois,'calibrationfiles',calibrationfiles,...
  'targets',targets,'f0',f0,'f1',f1);
