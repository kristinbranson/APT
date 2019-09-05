function feature = ConstructLandmarkFeature(featuretype,transtype,coordsys)

if strcmp(transtype,'none'),
  featurename = featuretype;
else
  featurename = [featuretype,'_',transtype];
end

feature = struct('name',featurename,'code',featurename,...
  'feature',featuretype,'transform',transtype,...
  'coordsystem',coordsys);
