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

t0 = find(any(any(~isnan(lpos),1),2),1,'first');
t1 = find(any(any(~isnan(lpos),1),2),1,'last');
nfrm = size(lpos,3);
trx = initializeTrx(lpos,lpostag,bodytrx,t0,t1);

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
    dmat2 = padData(dmat1,t0,t1,nfrm);
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
  dmat2 = padData(dmat2,t0,t1,nfrm);
end
fprintf('Time to compute info statistic %s = %f\n',prop.name,toc);

function trx = initializeTrx(lpos,occluded,bodytrx,t0,t1)

if nargin < 4,
  t0 = 1;
end
if nargin < 5,
  t1 = size(lpos,3);
end


trx = struct;
trx.pos = lpos(:,:,t0:t1);
trx.bodytrx = cropBodyTrx(bodytrx,t0,t1);
trx.occluded = double(occluded(:,t0:t1));
trx.realunits = false;
trx.pxpermm = [];
trx.fps = [];
trx.t0 = t0;
trx.t1 = t1;
trx.nfrm = size(lpos,3);

function bodytrx2 = cropBodyTrx(bodytrx,t0,t1)

if isempty(bodytrx),
  bodytrx2 = [];
  return;
end
fnscrop = intersect(fieldnames(bodytrx),{'x','y','theta','a','b','timestamps'});
bodytrx2 = struct;
i0 = t0 + bodytrx.off;
i1 = t1 + bodytrx.off;
for i = 1:numel(fnscrop),
  fn = fnscrop{i};
  bodytrx2.(fn) = bodytrx.(fn)(i0:i1);
end
bodytrx2.off = 0;
bodytrx2.firstframe = 1;
bodytrx2.endframe = t1-t0+1;
if isfield(bodytrx,'pxpermm'),
  bodytrx2.pxpermm = bodytrx.pxpermm;
end

function data = padData(dat,t0,t1,nfrm)

sz = size(dat.data);
data = cat(2,nan(sz(1),t0-1),dat.data,nan(sz(1),nfrm-t1));