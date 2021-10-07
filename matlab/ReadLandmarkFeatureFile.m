function features = ReadLandmarkFeatureFile(filename)

rawpropinfo = yaml.ReadYaml(filename);

features = EmptyLandmarkFeatureArray();

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
    features(end+1,1) = ConstructLandmarkFeature(featuretype,...
      transtypes{j},coordsystem); %#ok<AGROW>
  end
  
end