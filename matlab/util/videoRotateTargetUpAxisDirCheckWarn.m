function videoRotateTargetUpAxisDirCheckWarn(handles)

ax = handles.axes_curr;
if (strcmp(ax.XDir,'reverse') || strcmp(ax.YDir,'normal')) && ...
    handles.labeler.movieRotateTargetUp
  warningNoTrace('LabelerGUI:axDir',...
    'Main axis ''XDir'' or ''YDir'' is set to be flipped and .movieRotateTargetUp is set. Graphics behavior may be unexpected; proceed at your own risk.');
end

