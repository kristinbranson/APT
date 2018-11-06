function [dmat2,units] = ComputeLandmarkFeatureFromPos(lpos,lpostag,bodytrx,prop)
% lpos: [npts x 2 x nfrm x ntgt] label array as in
%   lObj.labeledpos{iMov}
% lpostag: [npts x nfrm x ntgt] logical as in lObj.labeledpostag{iMov}
% pcode: name/id of data to extract
%
% dmat: [npts x nfrm] data matrix for pcode, extracted from lpos

% TODO: this currently computes for all frames, even those that have
% not been labeled
tic;
units = parseunits('');

trx = initializeTrx(lpos,lpostag,bodytrx);



if isfield(trx,prop.code),
  dmat2 = trx.(prop.code);
  if isstruct(dmat2),
    dmat2 = dmat2.data;
  end
  fprintf('Time to compute info statistic %s = %f\n',prop.name,toc);
else
  
  if isfield(trx,prop.feature),
    dmat1 = trx.(prop.feature);
  else
    
    fun = sprintf('compute_landmark_%s',prop.feature);
    if ~exist(fun,'file'),
      warningNoTrace('Unknown property to display in timeline.');
      dmat2 = nan(size(lpos,1),size(lpos,3));
      fprintf('Time to compute info statistic %s = %f\n',prop.name,toc);
      return;
    end
    
    if strcmpi(prop.coordsystem,'Body'),
      
      trx = ComputeRelativeLandmarkPos_Body(trx);
      
    end
    
    trx = feval(fun,trx);
    dmat1 = trx.(prop.feature);
    
  end
  
  if strcmpi(prop.transform,'none'),
    units = dmat1.units;
    dmat2 = dmat1.data;
    fprintf('Time to compute info statistic %s = %f\n',prop.name,toc);
    return;
  end
  
  fun = sprintf('compute_landmark_transform_%s',prop.transform);
  if ~exist(fun,'file'),
    warningNoTrace('Unknown property to display in timeline.');
    dmat2 = nan(size(lpos,1),size(lpos,3));
    fprintf('Time to compute info statistic %s = %f\n',prop.name,toc);
    return;
  end
  
  dmat2 = feval(fun,dmat1);
  units = dmat2.units;
  dmat2 = dmat2.data;
end
fprintf('Time to compute info statistic %s = %f\n',prop.name,toc);

function trx = initializeTrx(lpos,occluded,bodytrx)

trx = struct;
trx.pos = lpos;
trx.bodytrx = bodytrx;
trx.occluded = double(occluded);
trx.realunits = false;
trx.pxpermm = [];
trx.fps = [];
