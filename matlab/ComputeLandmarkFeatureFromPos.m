function [dmat2,units] = ComputeLandmarkFeatureFromPos(...
    lpos,lpostag,t0,t1,nfrmtot,bodytrx,prop)
% Currently assumes single-target
%
% lpos: [npts x 2 x nfrmlpos] label array. (nfrmlpos=t1-t0)
% lpostag: [npts x nfrmlpos] logical 
% t0: absolute time labeling lpos(:,:,1)
% t1: absolute time labeling lpos(:,:,end). Note t1 is *not* "1-past"
% nfrmtot: total number of frames. The output is currently padded to this size
% bodytrx: trx from body tracking
% pcode: name/id of data to extract
%
% dmat2: [npts x nfrmtot] data matrix for pcode, extracted from lpos,
%   nan-padded if necessary

% TODO: this currently computes for all frames, even those that have
% not been labeled

tic;
units = parseunits('');

% t0 = find(any(any(~isnan(lpos),1),2),1,'first');
% t1 = find(any(any(~isnan(lpos),1),2),1,'last');
% nfrm = size(lpos,3);

trx = initializeTrx(lpos,lpostag,bodytrx,t0,t1,nfrmtot);

if isfield(trx,prop.code), 
  dmat2 = trx.(prop.code);
  if isstruct(dmat2),
    dmat2 = dmat2.data;
    % AL: should units be dmat2.units?
  end

  % AL: not sure how we would currently enter this branch; but thinking if
  % we got here all trx props should be 'cropped' to size t0:t1 so we would 
  % need to pad.
  % dmat2 = padData(dmat2,t0,t1,nfrm);
  % units = dmat2.units

%   fprintf('Time to compute info statistic %s = %f\n',prop.name,toc);
else
  propfeatr = prop.feature;
  proptrans = prop.transform;

  if ~isfield(trx,propfeatr),
    fun = sprintf('compute_landmark_%s',propfeatr);
    if ~is_valid_function_name(fun),
      warningNoTrace('Unknown property to display in timeline: %s.', fun);
      dmat2 = nan(size(lpos,1),nfrmtot);
%       fprintf('Time to compute info statistic %s = %f\n',prop.name,toc);
      return;
    end
    
    if strcmpi(prop.coordsystem,'Body'),      
      trx = ComputeRelativeLandmarkPos_Body(trx);      
    end
    
    trx = feval(fun,trx);
  end
  dmat1 = trx.(propfeatr);    
  
  if strcmpi(proptrans,'none'),
    units = dmat1.units;
    dmat2 = padData(dmat1,t0,t1,nfrmtot);
%     fprintf('Time to compute info statistic %s = %f\n',prop.name,toc);
    return;
  end
  
  fun = sprintf('compute_landmark_transform_%s',proptrans);
  if ~is_valid_function_name(fun),
    warningNoTrace('Unknown property to display in timeline: %s.', fun);
    dmat2 = nan(size(lpos,1),nfrmtot);
%     fprintf('Time to compute info statistic %s = %f\n',prop.name,toc);
    return;
  end
  
  dmat2 = feval(fun,dmat1);
  units = dmat2.units;
  dmat2 = padData(dmat2,t0,t1,nfrmtot);
end
% fprintf('Time to compute info statistic %s = %f\n',prop.name,toc);

function trx = initializeTrx(lpos,occluded,bodytrx,t0,t1,nfrmtot)

% if nargin < 4,
%   t0 = 1;
% end
% if nargin < 5,
%   t1 = size(lpos,3);
% end

trx = struct;
trx.pos = lpos;
trx.bodytrx = cropBodyTrx(bodytrx,t0,t1);
trx.occluded = double(occluded);
trx.realunits = false;
trx.pxpermm = [];
trx.fps = [];
trx.t0 = t0;
trx.t1 = t1;
trx.nfrm = nfrmtot;

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
