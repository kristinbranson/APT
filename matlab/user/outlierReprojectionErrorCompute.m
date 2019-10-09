function [suspscore,tblsusp,diagstr] = outlierReprojectionErrorCompute(lObj)
% Consider reprojection error of labeled frames
%
% suspscore: Maximum RP err (across points, views) for a frame, in px

if lObj.nview~=2
  error('Reprojection error currently only supported for stereo-view projects.');
end
if lObj.nTargets>1
  % All multiview projects currently are single-target
  error('Unsupported for multiple targets.');
end

resp = inputdlg('Specify maximum reprojection error threshold (px)','',1,{'2'});
if isempty(resp)
  suspscore = [];
  tblsusp = [];
  diagstr = [];
  return;
end
maxrpeThresh = str2double(resp{1});

fprintf(1,'Computing...\n');

nphyspts = lObj.nPhysPoints;
nvw = lObj.nview;
d = 2;
nmov = lObj.nmovies;

vcdPW = lObj.viewCalProjWide;
if vcdPW
  cr = lObj.viewCalibrationData; 
  assert(isa(cr,'CalRig'));
else
  vcd = lObj.viewCalibrationData;
  szassert(vcd,[nmov 1]);
end

tLbls = lObj.labelGetMFTableLabeled;

ptvwLbls = arrayfun(@(x,y)sprintf('pt%dvw%d',x,y),...
  [1:nphyspts 1:nphyspts]',[ones(1,nphyspts) 2*ones(1,nphyspts)]',...
  'uni',0);

suspscore = cell(nmov,1);
tblsusp = struct('mov',cell(0,1),'frm',[],'iTgt',[],'susp',[],'suspPt',[]);
for imov=1:nmov
  % lposI = lpos{imov}; % [npts x 2 x nfrm]
  if ~vcdPW
    cr = vcd{imov};
  end

  nfrms = lObj.movieInfoAll{imov,1}.nframes;
  maxrperr = nan(nfrms,1);

  irows = find(tLbls.mov==imov);
  frmslbl = tLbls.frm(irows);
  nfrmslbl = numel(frmslbl);
  if nfrmslbl==0
    suspscore{imov} = maxrperr;
    continue;
  else
    fprintf(1,'Mov %d, %d labeled frames.\n',imov,nfrmslbl);
  end
  
  p = tLbls.p(irows,:);
  p = reshape(p,[nfrmslbl nphyspts nvw d]);
  p = permute(p,[4 1 2 3]); % x/y,i,iphyspt,vw
  p = reshape(p,[2 nfrmslbl*nphyspts nvw]);
    
  % see iss #60,#245
  if isa(cr,'CalRigMLStro')
    [~,~,~,rperr1,rperr2] = cr.stereoTriangulate(p(:,:,1),p(:,:,2));  
  elseif isa(cr,'OrthoCamCalPair')
    [~,~,~,~,rperr1,rperr2] = cr.stereoTriangulate(p(:,:,1),p(:,:,2));
  elseif isa(cr,'CalRig2CamCaltech')
    y1 = p([2 1],:,1)';
    y2 = p([2 1],:,2)';
    [X1,X2] = cr.stereoTriangulateLR(y1,y2);
    szassert(X1,[3 nfrmslbl*nphyspts]);
    szassert(X2,[3 nfrmslbl*nphyspts]);
    xp1rp = cr.project(X1,'L');
    xp2rp = cr.project(X2,'R');
    y1rp = cr.x2y(xp1rp,'L');
    y2rp = cr.x2y(xp2rp,'R');
    
    rperr1 = sqrt(sum((y1-y1rp).^2,2))';
    rperr2 = sqrt(sum((y2-y2rp).^2,2))';    
  else
    assert(false,'Unsupported calibration object.');
  end 
  
  szassert(rperr1,[1 nfrmslbl*nphyspts]);
  szassert(rperr2,[1 nfrmslbl*nphyspts]);
  rperrfrmslbl = cat(2,reshape(rperr1,[nfrmslbl nphyspts]), ...
                       reshape(rperr2,[nfrmslbl nphyspts]) );
  [rperrmxfrmslbl,idxfrmslbl] = max(rperrfrmslbl,[],2);
  
  maxrperr(frmslbl) = rperrmxfrmslbl;  
  suspscore{imov} = maxrperr;
  for ifl=1:nfrmslbl
    if rperrmxfrmslbl(ifl)>maxrpeThresh
      tblsusp(end+1,1).mov = imov; %#ok<AGROW>
      tblsusp(end).frm = frmslbl(ifl);
      tblsusp(end).iTgt = 1;
      tblsusp(end).susp = rperrmxfrmslbl(ifl);
      tblsusp(end).suspPt = ptvwLbls{idxfrmslbl(ifl)};
    end
  end
end

tblsusp = struct2table(tblsusp);
sortvars = {'mov' 'susp'};
sortmode = {'ascend' 'descend'};
tblsusp = sortrows(tblsusp,sortvars,sortmode);

diagstr = 'Reprojection Error';
