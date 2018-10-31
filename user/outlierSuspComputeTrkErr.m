function [suspscore,tblsusp,diagstr] = outlierSuspComputeTrkErr(lObj)

% Compare tracking to lbls for CURRENT MOV ONLY. Look for frames/targets
% where max_overlandmarks(tracking error) > threshold

resp = inputdlg('Specify tracking err threshold (px)','',1,{'3'});
if isempty(resp)
  suspscore = [];
  tblsusp = [];
  diagstr = [];
  return;
end
errThresh = str2double(resp{1});

npts = lObj.nLabelPoints;
nmov = lObj.nmovies;
%ntgt = lObj.nTargets;
imovcurr = lObj.currMovie;

suspscore = cell(nmov,1);
%tblsusp = struct('mov',cell(0,1),'frm',[],'iTgt',[],'susp',[],'suspPt',[]);
mov = nan(0,1);
frm = nan(0,1);
iTgt = nan(0,1);
susp = nan(0,1);
suspPt = nan(0,1);
for imov=1:nmov
  lposI = lObj.labeledpos{imov};
  [~,~,nfrm,ntgt] = size(lposI);
  if imov~=imovcurr
    suspscore{imov} = zeros(nfrm,ntgt);
  else
    tpos = lObj.tracker.getTrackingResultsCurrMovie();
    err = squeeze(sqrt(sum((tpos-lposI).^2,2))); % npt x nfrm x ntgt
    [maxerr,maxerrpt] = max(err,[],1); % max over landmarks
    maxerr = reshape(maxerr,nfrm,ntgt); % nfrm x ntgt
    maxerrpt = reshape(maxerrpt,nfrm,ntgt); % nfrm x ntgt
    suspscore{imov} = maxerr;
    
    for itgt=1:ntgt
      maxerrtgt = maxerr(:,itgt); % nfrm x 1
      maxerrpttgt = maxerrpt(:,itgt); % nfrm x 1
      fkeep = find(maxerrtgt>errThresh);
      nkeep = numel(fkeep);
      
      mov = [mov;repmat(imov,nkeep,1)];
      frm = [frm;fkeep];
      iTgt = [iTgt;repmat(itgt,nkeep,1)];
      susp = maxerrtgt(fkeep);
      suspPt = maxerrpttgt(fkeep);      
    end
  end
end

tblsusp = table(mov,frm,iTgt,susp,suspPt);
sortvars = {'mov' 'susp'};
sortmode = {'ascend' 'descend'};
tblsusp = sortrows(tblsusp,sortvars,sortmode);

diagstr = sprintf('errThresh=%.3f',errThresh);
