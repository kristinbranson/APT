function [tbls,medmindist,kvals,savename] = ComputePixelPrecisionTable(gtdata,varargin)

[kfracs,nets,legendnames,colors,exptype,...
  conddata,labeltypes,datatypes,...
  pttypes,annoterrdata,...
  savedir,savename,dosavefig,...
  threshmethod,threshprctile,modeli,...
  kvals] = myparse(varargin,'kfracs',.25:.25:2,...
  'nets',{},'legendnames',{},'colors',[],'exptype','exp',...
  'conddata',[],'labeltypes',{},'datatypes',{},...
  'pttypes',{},...
  'annoterrdata',[],...
  'savedir','.',...
  'savename','',...
  'dosavefig',false,...
  'threshmethod','medmindist',...
  'threshprctile',95,...
  'modeli',[],...
  'kvals',[]);

isshexp = startsWith(exptype,'SH');

if isempty(nets),
  assert(~isempty(gtdata));
  nets = fieldnames(gtdata);
end
nnets = numel(nets);

if isempty(modeli),
  modelicurr = numel(gtdata.(nets{1}));
else
  modelicurr = min(modeli,numel(gtdata.(nets{1})));
end

ndatapts = size(gtdata.(nets{1}){modelicurr}.labels,1);
if isempty(conddata),
  conddata = struct;
  conddata.data_cond = ones(ndatapts,1);
  conddata.label_cond = ones(ndatapts,1);
end
[ndata,nlandmarks,ndim] = size(gtdata.(nets{1}){modelicurr}.labels);
if isempty(pttypes),
  npttypes = nlandmarks;
end
if isempty(pttypes),
  pttypes = cell(npttypes,2);
  for i = 1:npttypes,
    pttypes{i,1} = num2str(i);
    pttypes{i,2} = i;
  end
end
npttypes = size(pttypes,1);

if isempty(conddata),
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

if isempty(legendnames),
  legendnames = nets;
end

if isempty(colors),
  colors = lines(nnets);
end

% distance to closest part in all labels
labelscurr = gtdata.(nets{1}){modelicurr}.labels;
minds = nan(size(labelscurr,1),nlandmarks);
for i = 1:size(labelscurr,1),
  d = squareform(pdist(reshape(labelscurr(i,:,:),[nlandmarks,ndim])));
  d(eye(nlandmarks)==1) = inf;
  minds(i,:) = min(d,[],1);
end
medmindist = median(minds(:));

threshprctileerr = nan(1,nnets);
for ndx = 1:nnets,

  iscpr = contains(nets{ndx},'cpr');

  idxcurr = true(size(conddata.data_cond));
  iall = find(strcmp(datatypes,'all'),1);
  if ~isempty(iall),
    idxcurr = idxcurr & ismember(conddata.data_cond,datatypes{iall,2});
  end
  iall = find(strcmp(labeltypes,'all'),1);
  if ~isempty(iall),
    idxcurr = idxcurr & ismember(conddata.label_cond,labeltypes{iall,2});
  end
  if iscpr && isshexp,
    % special case for SH/cpr whose computed GT output only has
    % 1149 rows instead of 1150 cause dumb
    idxcurr(4) = [];
  end
  
  if isempty(modeli),
    modelicurr = numel(gtdata.(nets{ndx}));
  else
    modelicurr = min(modeli,numel(gtdata.(nets{ndx})));
  end

  err = sqrt(sum((gtdata.(nets{ndx}){modelicurr}.labels(idxcurr,:,:)-gtdata.(nets{ndx}){modelicurr}.pred(idxcurr,:,:)).^2,3));
  %threshprctileerr(ndx) = prctile(err(:),threshprctile);
  maxerr = max(err,[],2);
  threshprctileerr(ndx) = prctile(maxerr,threshprctile);
end
minthreshprctileerr = min(threshprctileerr);

if isempty(kvals),
  switch threshmethod,
    case 'medmindist',
      kvals = medmindist*kfracs;
    case 'prctileerr',
      kvals = minthreshprctileerr*kfracs;
    otherwise
      error('unknown method for choosing thresholds %s',threshmethod);
  end
end

if ~isempty(annoterrdata),
  annfns = fieldnames(annoterrdata);
else
  annfns = {};
end
nann = numel(annfns);

% nnets x npttypes+1 x nks x ndatatypes x nlabeltypes
aperrk = nan([nnets+nann,npttypes+1,numel(kvals),ndatatypes,nlabeltypes]);
% nnets x nks x ndatatypes x nlabeltypes
apmaxerrk = nan([nnets+nann,numel(kvals),ndatatypes,nlabeltypes]);
apmnerrk = nan([nnets+nann,numel(kvals),ndatatypes,nlabeltypes]);
for ndx = 1:nnets+nann,
  if ndx > nnets,
    err = sqrt(sum((annoterrdata.(annfns{ndx-nnets}){end}.labels-annoterrdata.(annfns{ndx-nnets}){end}.pred).^2,3));
    conddatacurr = annoterrdata.(annfns{ndx-nnets}){end};
  else
    
    if isempty(modeli),
      modelicurr = numel(gtdata.(nets{ndx}));
    else
      modelicurr = min(modeli,numel(gtdata.(nets{ndx})));
    end
    
    err = sqrt(sum((gtdata.(nets{ndx}){modelicurr}.labels-gtdata.(nets{ndx}){modelicurr}.pred).^2,3));
    conddatacurr = conddata;
  end
  iscpr = ndx <= nnets && ~isempty(strfind(nets{ndx},'cpr'));
  
  for datai = 1:ndatatypes,
    for labeli = 1:nlabeltypes,
      
      idxcurr = ismember(conddatacurr.data_cond,datatypes{datai,2})&ismember(conddatacurr.label_cond,labeltypes{labeli,2});
      if iscpr && isshexp,
        % special case for SH/cpr whose computed GT output only has
        % 1149 rows instead of 1150 cause dumb
        idxcurr(4) = [];
      end

      err0 = err(idxcurr,:);
      maxerr = max(err0,[],2);
      mnerr = mean(err0,2);
      for kk = 1:numel(kvals),
        k = kvals(kk);
        aperrk(ndx,1,kk,datai,labeli) = nnz(err0<=kvals(kk))/numel(err0);
        apmaxerrk(ndx,kk,datai,labeli) = nnz(maxerr<=kvals(kk))/numel(maxerr);
        apmnerrk(ndx,kk,datai,labeli) = nnz(mnerr<=kvals(kk))/numel(mnerr);
        for pti = 1:npttypes,
          err1 = err0(:,pttypes{pti,2});
          aperrk(ndx,pti+1,kk,datai,labeli) = nnz(err1<=kvals(kk))/numel(err1);
        end
      end
    end
  end
end
% nnets x npttypes+1 x nks x ndatatypes x nlabeltypes

% nnets x npttypes+1 x ndatatypes x nlabeltypes
meanaperr = permute(mean(aperrk,3),[1,2,4,5,3]); 
% nnets x ndatatypes x nlabeltypes
meanapmaxerr = permute(mean(apmaxerrk,2),[1,3,4,2]);
meanapmnerr = permute(mean(apmnerrk,2),[1,3,4,2]);

fns = [legendnames(:);annfns];
  
if dosavefig,
  if isempty(savename),
    savename = fullfile(savedir,sprintf('appxdata_%s.tex',exptype));
  end
  fid = fopen(savename,'w');
else
  fid = 1;
end
fprintf(fid,'AP averaged over pixel thresholds = %s\\\\\n',sprintf('%.2f ',kvals));
fprintf(fid,'median inter-part distance = %.2f. minimum %.1fth percentile of worst landmark error = %.2f\\\\\n\n',medmindist,threshprctile,minthreshprctileerr);

tbls = cell(ndatatypes,nlabeltypes);
for datai = 1:ndatatypes,
  for labeli = 1:nlabeltypes,
    
    tbl = table;
    tbl.APall = meanaperr(:,1,datai,labeli);
    tbl.Properties.VariableDescriptions{1} = 'AP';
    tbl.Properties.RowNames = fns;
    for pti = 1:npttypes,
      tbl.(sprintf('AP_part%d',pti)) = meanaperr(:,1+pti,datai,labeli);
      tbl.Properties.VariableDescriptions{end} = sprintf('AP %s',pttypes{pti,1});
    end
    tbl.AWP = meanapmaxerr(:,datai,labeli);
    tbl.Properties.VariableDescriptions{end} = 'AWP';
    tbl.AMP = meanapmnerr(:,datai,labeli);
    tbl.Properties.VariableDescriptions{end} = 'AMP';    
    for k = 1:numel(kvals),
      tbl.(sprintf('P_k%d_all',k)) = aperrk(:,1,k,datai,labeli);
      tbl.Properties.VariableDescriptions{end} = sprintf('P/k=%.1f',kvals(k));
    end
    for k = 1:numel(kvals),
      for pti = 1:npttypes,
        tbl.(sprintf('P_k%d_part%d',k,pti)) = aperrk(:,1+pti,k,datai,labeli);
        tbl.Properties.VariableDescriptions{end} = sprintf('P/k=%.1f/%s',kvals(k),pttypes{pti,1});
      end
    end
    for k = 1:numel(kvals),
      tbl.(sprintf('WP_k%d',k)) = apmaxerrk(:,k,datai,labeli);
      tbl.Properties.VariableDescriptions{end} = sprintf('WP/k=%.1f',kvals(k));
    end
    tbl.Properties.Description = sprintf('%s/%s',datatypes{datai},labeltypes{labeli});
    tbls{datai,labeli} = tbl;
  end
end

for datai = ndatatypes,
  for labeli = 1:nlabeltypes,
    fprintf(fid,['\\begin{tabular}{|c||',repmat('c|',[1,nnets+numel(annfns)]),'}']);
    fprintf(fid,'\\hline\n');
    fprintf(fid,'Measure - %s',labeltypes{labeli});
    for i = 1:nnets+nann,
      fprintf(fid,' & %s',fns{i});
    end
    fprintf(fid,'\\\\\\hline\\hline\n');
    
    fprintf(fid,'AP');
    PrintTableLine(fid,meanaperr(:,1,datai,labeli),nnets);
    fprintf(fid,'AWP');
    PrintTableLine(fid,meanapmaxerr(:,datai,labeli),nnets);
    fprintf(fid,'AMP');
    PrintTableLine(fid,meanapmnerr(:,datai,labeli),nnets);

    for pti = 1:npttypes,
      fprintf(fid,'AP - %s',pttypes{pti,1});
      PrintTableLine(fid,meanaperr(:,pti+1,datai,labeli),nnets);
    end
    for k = 1:numel(kvals),
      fprintf(fid,'P/k=%.1f',kvals(k));
      PrintTableLine(fid,aperrk(:,1,k,datai,labeli),nnets);
    end
    for k = 1:numel(kvals),
      fprintf(fid,'WP/k=%.1f',kvals(k));
      PrintTableLine(fid,apmaxerrk(:,k,datai,labeli),nnets);
    end
%     for pti = 1:npttypes,
%       for k = 1:numel(kvals),
%         fprintf(fid,'AP@k=%.1f/%s',kvals(k),pttypes{pti,1});
%         for ndx = 1:nnets+numel(annfns),
%           fprintf(fid,' & %.2f',aperrk(ndx,pti+1,k,datai,labeli));
%         end
%         fprintf(fid,'\\\\\\hline\n');
%       end
%     end
    fprintf(fid,'\\end{tabular}\\\\\n\n');
  end
end
if fid > 1,
  fclose(fid);
end

function PrintTableLine(fid,x,n1)

if nargin < 3,
  n1 = numel(x);
end

bolds10 = '{\bf ';
bolds20 = '}';

roundx = round(100*x)/100;
maxx = max(roundx(1:n1));
isbest = roundx >= maxx;
n = numel(x);
for ndx = 1:n,
  if isbest(ndx),
    bolds1 = bolds10;
    bolds2 = bolds20;
  else
    bolds1 = '';
    bolds2 = '';
  end
  fprintf(fid,' & %s%.2f%s',bolds1,x(ndx),bolds2);
end
fprintf(fid,'\\\\\\hline\n');

