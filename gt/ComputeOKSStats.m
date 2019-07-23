function [apk,ap,meanoks,apvals] = ComputeOKSStats(data,kappa,varargin)

[conddata,pttypes,datatypes,labeltypes,apvals,meanapvals,distname,netname,exptype] = myparse(varargin,'conddata',[],'pttypes',{},...
  'datatypes',{},'labeltypes',{},'apvals',[40,50,60],'meanapvals',30:5:70,'distname','gamma2','netname','','exptype','');

isshexp = startsWith(exptype,'SH');
iscpr = contains(netname,'cpr');%~isempty(strfind(netname,'cpr'));

[ndatapts,npts,d] = size(data.labels);
if isempty(pttypes),
  pttypes = cell(npts,2);
  for i = 1:npts,
    pttypes{i,1} = num2str(i);
    pttypes{i,2} = i;
  end
end
npttypes = size(pttypes,1);

if isempty(conddata),
  conddata = struct;
  conddata.data_cond = ones(ndatapts,1);
  conddata.label_cond = ones(ndatapts,1);
  ndatatypes = 1;
  nlabeltypes = 1;
else
  ndatatypes = max(conddata.data_cond);
  nlabeltypes = max(conddata.label_cond);
end
if isempty(labeltypes),
  labeltypes = cell(nlabeltypes,2);
  for i = 1:nlabeltypes,
    labeltypes{i,1} = num2str(i);
    labeltypes{i,2} = i;
  end
end
if isempty(datatypes),
  datatypes = cell(ndatatypes,2);
  for i = 1:ndatatypes,
    datatypes{i,1} = num2str(i);
    datatypes{i,2} = i;
  end
end
ndatatypes = size(datatypes,1);
nlabeltypes = size(labeltypes,1);

[ks,oks,okstype] = ComputeOKS(data,kappa,'pttypes',pttypes,'distname',distname);

meanoks = nan(npttypes+1,ndatatypes,nlabeltypes);
napvals = numel(apvals);
apk = nan([napvals,npttypes+1,ndatatypes,nlabeltypes]);
ap = zeros([npttypes+1,ndatatypes,nlabeltypes]);

for datai = 1:ndatatypes,
  for labeli = 1:nlabeltypes,
    idxcurr = ismember(conddata.data_cond,datatypes{datai,2})&ismember(conddata.label_cond,labeltypes{labeli,2});
    if iscpr && isshexp,
      % special case for SH/cpr whose computed GT output only has
      % 1149 rows instead of 1150 cause dumb
      idxcurr(4) = [];
    end
    
    meanoks(1,datai,labeli) = mean(oks(idxcurr));
    meanoks(2:end,datai,labeli) = mean(okstype(idxcurr,:),1);
    for k = 1:napvals,
      apk(k,1,datai,labeli) = nnz(oks(idxcurr)>=apvals(k)/100)/nnz(idxcurr);
      apk(k,2:end,datai,labeli) = sum(okstype(idxcurr,:)>=apvals(k)/100,1)/nnz(idxcurr);
    end
    
    for k = 1:numel(meanapvals),
      ap(1,datai,labeli) = ap(1,datai,labeli) + nnz(oks(idxcurr)>=meanapvals(k)/100)/nnz(idxcurr);
      ap(2:end,datai,labeli) = ap(2:end,datai,labeli) + sum(okstype(idxcurr,:)>=meanapvals(k)/100,1)'/nnz(idxcurr);
    end
    ap(:,datai,labeli) = ap(:,datai,labeli) / numel(meanapvals);

  end
end