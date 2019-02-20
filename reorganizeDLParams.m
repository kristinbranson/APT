function s = reorganizeDLParams(s)

if isfield(s,'trackParams'),
  return;
end

% KB 20190212: reorganized DL parameters -- many specific parameters
% were moved to common, and organized common parameters
sPrm_common = APTParameters.defaultParamsStructDTCommon;
fns1 = fieldnames(sPrm_common);
fns0 = fieldnames(s.trackDLParams);
tfCommonMatch = isempty(setxor(fns0,fns1));

sPrm_common_in = flattenStruct(s.trackDLParams);
leaves_common = fieldnames(sPrm_common_in);
ts = 0;
ts_common = [];
for i = 1:numel(leaves_common),
  fn = leaves_common{i};
  [sPrm_common,nseti,ts_common] = setStructLeaf(sPrm_common,fn,sPrm_common_in.(fn),'ts_set',ts,'ts_struct',ts_common);
  assert(nseti <= 1);
end

ts_all = zeros(1,numel(s.trackerData));
for i = 1:numel(s.trackerData),
  if ~isfield(s.trackerData{i},'trnName') || isempty(s.trackerData{i}.trnName),
    continue;
  end
  ts_all(i) = datenum(s.trackerData{i}.trnName,'yyyymmddTHHMMSS');
end

[~,order] = sort(ts_all,'descend');

tfCollapse = false;
for i = order,
  if ~strcmp(s.trackerClass{i}{1},'DeepTracker'),
    continue;
  end
  if isempty(s.trackerData{i}),
    continue;
  end
  
  sPrm_specific = APTParameters.defaultParamsStructDT(char(s.trackerData{i}.trnNetType));
  tfSpecificParams = ~isempty(sPrm_specific);
  
  % check if parameter names are up-to-date
  if tfSpecificParams,
    fns1 = fieldnames(sPrm_specific);
    fns0 = fieldnames(s.trackerData{i}.sPrm);
    if tfCommonMatch && isempty(setxor(fns0,fns1)),
      continue;
    end
  end
  tfCollapse = true;
  
  sPrm_specific_in = flattenStruct(s.trackerData{i}.sPrm);
  leaves_specific = fieldnames(sPrm_specific_in);
  if ~isfield(s.trackerData{i},'trnName') || isempty(s.trackerData{i}.trnName),
    ts = 0;
  else
    ts = datenum(s.trackerData{i}.trnName,'yyyymmddTHHMMSS');
  end
  
  warnfun_common = @(fn) sprintf('Collision collapsing from %s to common DL parameter %s, using most recent value',char(s.trackerData{i}.trnNetType),fn);
  warnfun_specific = @(fn) sprintf('Collision collapsing %s parameter %s, using most recent value',char(s.trackerData{i}.trnNetType),fn);
  ts_specific = [];
  for j = 1:numel(leaves_specific),
    fn = leaves_specific{j};
    if tfSpecificParams,
      [sPrm_specific,nsetj,ts_specific] = setStructLeaf(sPrm_specific,fn,sPrm_specific_in.(fn),'ts_set',ts,'ts_struct',ts_specific,'warnfun',warnfun_specific);
      assert(nsetj <= 1);
      if nsetj == 1,
        continue;
      end
    end
    [sPrm_common,nsetj,ts_common] = setStructLeaf(sPrm_common,fn,sPrm_specific_in.(fn),'ts_set',ts,'ts_struct',ts_common,'warnfun',warnfun_common);
    if(nsetj ~= 1),
      warningNoTrace(sprintf('Parameter %s stored in %s parameters obsolete, ignoring.',fn,char(s.trackerData{i}.trnNetType)));
    end
  end
  if tfSpecificParams,
    ts_specific_flat = flattenStruct(ts_specific);
    fns = fieldnames(ts_specific_flat);
    for j = 1:numel(fns),
      if ts_specific_flat.(fns{j}) < 0,
        warningNoTrace(sprintf('Could not find %s parameter %s in loaded project, using default value',char(s.trackerData{i}.trnNetType),fns{i}));
      end
    end
  end
  s.trackerData{i}.sPrm = sPrm_specific;
  
end

if tfCollapse,
  % check that all parameters were set
  ts_common_flat = flattenStruct(ts_common);
  fns = fieldnames(ts_common_flat);
  for i = 1:numel(fns),
    if ts_common_flat.(fns{i}) < 0,
      warningNoTrace(sprintf('Could not find common DL parameter %s in loaded project, using default value',fns{i}));
    end
  end
end

s.trackDLParams = sPrm_common;