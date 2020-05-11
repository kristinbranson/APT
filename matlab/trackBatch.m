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

% call tracker.track to do the real tracking
lObj.tracker.track(movfiles,'trxfiles',trxfiles,'trkfiles',trkfiles,...
  'cropRois',cropRois,'calibrationfiles',calibrationfiles,...
  'targets',targets,'f0',f0s,'f1',f1s);
