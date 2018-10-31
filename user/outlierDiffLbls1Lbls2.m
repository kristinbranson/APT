function [suspscore,tblsusp,diagstr] = outlierDiffLbls1Lbls2(lObj)

resp = inputdlg('Specify mean pixel err threshold','',1,{'4.0'});
if isempty(resp)
  suspscore = [];
  tblsusp = [];
  diagstr = [];
  return;
end
dlThresh = str2double(resp{1});
  

npts = lObj.nLabelPoints;
nmov = lObj.nmovies;
lpos = lObj.labeledpos;
lpos2 = lObj.labeledpos2; % Imported tracking labels/positions
dlpos = cellfun(@(x,y) squeeze(sqrt(sum((x-y).^2,2))),lpos,lpos2,'uni',0);
dlposmean = cellfun(@(x)nanmean(x,1),dlpos,'uni',0);

suspscore = cell(nmov,1);
tblsusp = struct('mov',cell(0,1),'frm',[],'iTgt',[],'susp',[],'suspPt',[]);
mov = nan(0,1);
frm = nan(0,1);
iTgt = nan(0,1);
susp = nan(0,1);
suspPt = nan(0,1);
for imov=1:nmov
  dl = dlposmean{imov}; % [1xnfrm]  
  dl = dl(:);
  fkeep = find(dl>dlThresh);
  nkeep = numel(fkeep);
  
  suspscore{imov} = dl;    

  dlfull = dlpos{imov}; % [npt x nfrm];
  dlfullkeep = dlfull(:,fkeep); % [npt x nkeep];
  [~,newsusppt] = max(dlfullkeep,[],1);
  
  mov = [mov; repmat(imov,nkeep,1)];
  frm = [frm; fkeep(:)];
  iTgt = [iTgt; ones(nkeep,1)];
  susp = [susp; dl(fkeep)];
  
  suspPt = [suspPt; newsusppt(:)];  
end

tblsusp = table(mov,frm,iTgt,susp,suspPt);
sortvars = {'mov' 'susp'};
sortmode = {'ascend' 'descend'};
tblsusp = sortrows(tblsusp,sortvars,sortmode);

diagstr = sprintf('diffThresh=%.3f',dlThresh);
