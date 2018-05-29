function [suspscore,tblsusp,diagstr] = outlierSuspCompute(lObj)

resp = inputdlg('Specify z-score threshold','',1,{'2.5'});
if isempty(resp)
  suspscore = [];
  tblsusp = [];
  diagstr = [];
  return;
end
zsThresh = str2double(resp{1});
  

npts = lObj.nLabelPoints;
nmov = lObj.nmovies;
lpos = lObj.labeledpos2; % Imported tracking labels/positions

% Compute SD of x,y,dx,dy for all pts
dlpos = cellfun(@(x)diff(x,1,3),lpos,'uni',0); % diff along frame dim
lposMega = cat(3,lpos{:}); % {all pts} x {x,y} x {all frames across movs}
dlposMega = cat(3,dlpos{:}); % {all pts} x {dx,dy} x {all consecutive-frame-pairs across movs}
ntotfrm = size(lposMega,3);
szassert(lposMega,[npts 2 ntotfrm]);
szassert(dlposMega,[npts 2 ntotfrm-nmov]);
% All these are [nptsx2]
lposMu = mean(lposMega,3,'omitnan'); 
dlposMu = mean(dlposMega,3,'omitnan');
lposSd = std(lposMega,0,3,'omitnan');
dlposSd = std(dlposMega,0,3,'omitnan');

% Compute z-scores
suspscore = cell(nmov,1);
tblsusp = struct('mov',cell(0,1),'frm',[],'iTgt',[],'susp',[],'suspPt',[]);
for imov=1:nmov
  lposI = lpos{imov}; % [npts x 2 x nfrm]
  dlposI = dlpos{imov}; % [npts x 2 x (nfrm-1)]
  
  zsLposI = abs(bsxfun(@rdivide,bsxfun(@minus,lposI,lposMu),lposSd));
  zsDlposI = abs(bsxfun(@rdivide,bsxfun(@minus,dlposI,dlposMu),dlposSd));
  
  nfrm = size(lposI,3);
  szassert(zsLposI,[npts 2 nfrm]);
  szassert(zsDlposI,[npts 2 nfrm-1]);
  zsLposILbl = cell(npts,2);
  zsDlposILbl = cell(npts,2);
  COORD = {'x' 'y'};
  for ipt=1:npts
    for j=1:2
      coord = COORD{j};
      zsLposILbl{ipt,j} = sprintf('pt%d/%s',ipt,coord);
      zsDlposILbl{ipt,j} = sprintf('pt%d/d%s',ipt,coord);
    end
  end
  zsLposI = reshape(zsLposI,[npts*2 nfrm]);
  zsDlposI = reshape(zsDlposI,[npts*2 nfrm-1]);
  zsDlposI(:,end+1) = zsDlposI(:,end);
  zsBigI = [zsLposI;zsDlposI]; % [npts*4 x nfrm]
  zsBigILbl = [zsLposILbl(:);zsDlposILbl(:)]; % [npts*4]

  [zsMax,idx] = max(zsBigI,[],1);
  % maximum zscore across {all pts}, {x and y}, {pos and vel}

  suspscore{imov} = zsMax(:);
  for f=1:nfrm
    if zsMax(f)>zsThresh
      tblsusp(end+1,1).mov = imov; %#ok<AGROW>
      tblsusp(end).frm = f;
      tblsusp(end).iTgt = 1;
      tblsusp(end).susp = zsMax(f);
      tblsusp(end).suspPt = zsBigILbl{idx(f)};
    end
  end
end

tblsusp = struct2table(tblsusp);
sortvars = {'mov' 'susp'};
sortmode = {'ascend' 'descend'};
tblsusp = sortrows(tblsusp,sortvars,sortmode);

diagstr = sprintf('zsThresh=%.3f',zsThresh);
