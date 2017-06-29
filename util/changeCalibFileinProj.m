function changeCalibFileinProj(PROJFILES)
%%changes calibration file used by all videos in a project
%
%
% PROJFILES = cell array containg strings with full paths and filenames of projects to replace
% calibration file for.
%
% newCalibFile = string containg full path and filename of new calibration
% file to use for PROJFILEDS.

%set this to new calibration file to use
occp = load(newCalibFile);
occp = occp.calObj;
%%
for iProj=1:numel(PROJFILES)
  proj = PROJFILES{iProj};
  lbl = load(proj,'-mat');
  assert(~lbl.viewCalProjWide);
  nMov = numel(lbl.labeledpos);
  assert(iscell(lbl.viewCalibrationData) && numel(lbl.viewCalibrationData)==nMov);
  for iMov=1:nMov
    lbl.viewCalibrationData{iMov} = occp;
  end  
  save(proj,'-struct','lbl'); % OVERWRITES projfile
  fprintf(1,'Done with %s\n',proj);
end