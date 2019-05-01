function [suspscore,tblsusp,diagstr] = outlierDiffTrkRes(obj)

if obj.nTargets>1
  error('Multiple targets currently unsupported.');
end
  
trIDsAll = obj.trkResIDs;
if numel(trIDsAll)<2
  error('Project does not have two sets of tracking results loaded.');
end


trIDstr = String.cellstr2CommaSepList(trIDsAll);
prompt = {
  sprintf('Specify tracking result set 1. Choices: %s',trIDstr);
  'Specify tracking result set 2.'
  };
resp = inputdlg(prompt,'Tracking Comparison',1,trIDsAll(1:2));
if isempty(resp)
  suspscore = [];
  tblsusp = [];
  diagstr = [];
  return;
end
assert(iscellstr(resp) && numel(resp)==2);

[tf,iTRs] = cellfun(@obj.trackResFindID,resp);
assert(all(tf));

resp = inputdlg('Specify mean pixel err threshold','',1,{'4.0'});
if isempty(resp)
  suspscore = [];
  tblsusp = [];
  diagstr = [];
  return;
end
dlThresh = str2double(resp{1});

npts = obj.nLabelPoints;
nmov = obj.nmovies;
trkres = obj.trkRes(:,:,iTRs);
tftrkres = ~cellfun(@isempty,trkres);
tftrkres = squeeze(all(tftrkres,2)); % views should either all have trkres or not
szassert(tftrkres,[nmov 2]);

dlpos = cell(nmov,1); % dlpos{imov} will be [npt x nfrm(imov)]
dlposmean = cell(nmov,1); % dlposmean{imov} will be [1 x nfrm(imov)]
for imov=1:nmov
  nfrm = obj.movieInfoAll{imov,1}.nframes;
  if all(tftrkres(imov,:))
    trkresI = trkres(imov,:,:); % [1 x nvw x 2]
    pTrk = cellfun(@(x)x.pTrk,trkresI,'uni',0);
    pTrk1 = cat(1,pTrk{1,:,1});
    pTrk2 = cat(1,pTrk{1,:,2});
    szassert(pTrk1,[npts 2 nfrm]);
    szassert(pTrk2,[npts 2 nfrm]);
    dlpos{imov} = squeeze(sqrt(sum((pTrk1-pTrk2).^2,2))); % [npts x nfrm]
    dlposmean{imov} = nanmean(dlpos{imov},1);
  else
    if any(tftrkres(imov,:))
      warningNoTrace('Movie %d has tracking results for one set but not the other and will not be included.',imov);
    end    
    dlpos{imov} = nan(npts,nfrm);
    dlposmean{imov} = nan(1,nfrm);
  end
end

suspscore = cell(nmov,1);
mov = nan(0,1);
frm = nan(0,1);
iTgt = nan(0,1);
susp = nan(0,1);
suspPt = nan(0,1);
for imov=1:nmov
  dl = dlposmean{imov}; % [1xnfrm]. L2 diff aved over all pts
  dl = dl(:);
  fkeep = find(dl>dlThresh);
  nkeep = numel(fkeep);
  
  suspscore{imov} = dl;    

  dlfull = dlpos{imov}; % [npt x nfrm];
  dlfullkeep = dlfull(:,fkeep); % [npt x nkeep];
  [~,newsusppt] = max(dlfullkeep,[],1);
  
  mov = [mov; repmat(imov,nkeep,1)]; %#ok<AGROW>
  frm = [frm; fkeep(:)]; %#ok<AGROW>
  iTgt = [iTgt; ones(nkeep,1)]; %#ok<AGROW>
  susp = [susp; dl(fkeep)]; %#ok<AGROW>
  
  suspPt = [suspPt; newsusppt(:)]; %#ok<AGROW>
end

tblsusp = table(mov,frm,iTgt,susp,suspPt);
sortvars = {'mov' 'susp'};
sortmode = {'ascend' 'descend'};
tblsusp = sortrows(tblsusp,sortvars,sortmode);

diagstr = sprintf('diffThresh=%.3f',dlThresh);
