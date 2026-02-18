function result = tidyToTrackStructForBatchTracking(toTrack)
% Patch up the toTrack struct to make it nice.

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
if ~isfield(toTrack,'targets')
  toTrack.targets = {};
elseif ~iscell(toTrack.targets) && size(toTrack.movfiles,1) == 1,
  toTrack.targets = {toTrack.targets};
end

if ~isfield(toTrack,'trxfiles')
  toTrack.trxfiles = {};
end

if ~isfield(toTrack,'calibrationfiles') || isempty(toTrack.calibrationfiles),
  calibrationfiles = {};
elseif ischar(toTrack.calibrationfiles),
  calibrationfiles = {toTrack.calibrationfiles};
else
  calibrationfiles = toTrack.calibrationfiles;
end
assert(iscell(calibrationfiles));

% Create the output struct
result = toTrack ;
result.f0s = f0s ;
result.f1s = f1s ;
result.calibrationfiles = calibrationfiles ;

end