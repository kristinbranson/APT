function APTCluster(varargin)
% APTCluster(varargin)
%
% APTCluster(lblFile,'retrain')
% 
% Do a full retrain of a project's tracker.

lblFile = varargin{1};
action = varargin{2};
if exist(lblFile,'file')==0
  error('APTCluster:file','Cannot find project file: ''%s''.',lblFile);
end
fprintf('APTCluster: ''%s'' on project ''%s''.\n',action,lblFile);

lObj = Labeler();
lObj.projLoad(lblFile);

switch action
  case 'retrain'
    lObj.trackRetrain();
    [p,f,e] = fileparts(lblFile);
    outfile = fullfile(p,[f '_retrain' datestr(now,'yyyymmddTHHMMSS') e]);
    fprintf('APTCluster: saving retrained project: %s\n',outfile);
    lObj.projSaveRaw(outfile);
  otherwise
    error('APTCluster:action','Unrecognized action ''%s''.',action);
end

delete(lObj);
