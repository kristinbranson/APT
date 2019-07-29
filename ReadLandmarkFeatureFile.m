function features = ReadLandmarkFeatureFile(filename)

rawpropinfo = ReadYaml(filename);

features = [];

featuretypes = fieldnames(rawpropinfo.Features);
for i = 1:numel(featuretypes),
  
  featuretype = featuretypes{i};
  x = rawpropinfo.Features.(featuretype);
  if isfield(x,'CoordSystem'),
    coordsystem = x.CoordSystem;
  else
    coordsystem = 'Global';
  end
  if isfield(x,'TransTypes'),
    transtypes = x.TransTypes;
  elseif isfield(x,'FeatureClass'),
    transtypes = rawpropinfo.FeatureClasses.(x.FeatureClass).TransTypes;
  else
    transtypes = {'none'};
  end
  
  for j = 1:numel(transtypes),

    if strcmp(transtypes{j},'none'),
      featurename = featuretype;
    else
      featurename = [featuretype,'_',transtypes{j}];
    end
    feature = struct('name',featurename,'code',featurename,...
      'feature',featuretype,'transform',transtypes{j},...
      'coordsystem',coordsystem);
    if isempty(features),
      features = feature;
    else
      features(end+1,1) = feature; %#ok<AGROW>
    end
  end
  
end