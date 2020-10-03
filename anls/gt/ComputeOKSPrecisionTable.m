function [kappa,apk,ap,meanoks,savename] = ComputeOKSPrecisionTable(gtdata,varargin)

[kappadistname,apvals,meanapvals,nets,legendnames,colors,exptype,...
  conddata,labeltypes,datatypes,...
  pttypes,annoterrdata,...
  savedir,savename,dosavefig,...
  threshmethod,threshprctile,...
  dormoutliers,nremove] = myparse(varargin,...
  'kappadistname','gamma2',...
  'apvals',[],'meanapvals',[],...
  'nets',{},'legendnames',{},'colors',[],'exptype','exp',...
  'conddata',[],'labeltypes',{},'datatypes',{},...
  'pttypes',{},...
  'annoterrdata',[],...
  'savedir','.',...
  'savename','',...
  'dosavefig',false,...
  'threshmethod','medmindist',...
  'threshprctile',95,...
  'dormoutliers',true,...
  'nremove',5);


if isempty(apvals),
  switch kappadistname
    case 'gaussian'
      apvals = [50,75];
    case 'gamma2'
      apvals = [30,40,50];
    otherwise
      error('not implemented');
  end
end
if isempty(meanapvals),
  switch kappadistname
    case 'gaussian'
      meanapvals = 50:5:95;
    case 'gamma2'
      meanapvals = 30:5:70;
    otherwise
      error('not implemented');
  end
end


if isempty(nets),
  assert(~isempty(gtdata));
  nets = fieldnames(gtdata);
end
nnets = numel(nets);

nets0 = intersect(nets,fieldnames(gtdata));

ndatapts = size(gtdata.(nets0{1}){end}.labels,1);
if isempty(conddata),
  conddata = struct;
  conddata.data_cond = ones(ndatapts,1);
  conddata.label_cond = ones(ndatapts,1);
end
nlandmarks = size(gtdata.(nets0{1}){end}.labels,2);
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

annfns = fieldnames(annoterrdata);
%kappadistname = 'gamma2';
[kappa,errs,areas,hfig] = TuneOKSKappa(annoterrdata,'distname',kappadistname,'pttypes',pttypes,'doplot',true,'dormoutliers',dormoutliers,'nremove',nremove);
set(hfig,'Units','pixels','Position',[10,10,560,1168]);
if dosavefig,
  saveas(hfig,fullfile(savedir,sprintf('IntraAnnotatorDistributionFit_%s_%s',kappadistname,exptype)),'svg');
  saveaspdf_JAABA(hfig,fullfile(savedir,sprintf('IntraAnnotatorDistributionFit_%s_%s.pdf',kappadistname,exptype)))
end
apk = cell(nnets+numel(annfns),1);
ap = cell(nnets+numel(annfns),1);
meanoks = cell(nnets+numel(annfns),1);

for i = 1:numel(annfns),
  [apk{nnets+i},ap{nnets+i},meanoks{nnets+i}] = ComputeOKSStats(annoterrdata.(annfns{i}){end},kappa,'pttypes',pttypes,...
    'conddata',annoterrdata.(annfns{i}){end},'pttypes',pttypes,'labeltypes',labeltypes,'datatypes',datatypes,...
    'apvals',apvals,'meanapvals',meanapvals,'distname',kappadistname,'netname',annfns{i},'exptype',exptype);
end
for ndx = 1:nnets,
  [apk{ndx},ap{ndx},meanoks{ndx}] = ComputeOKSStats(gtdata.(nets{ndx}){end},kappa,'pttypes',pttypes,...
    'conddata',conddata,'pttypes',pttypes,'labeltypes',labeltypes,'datatypes',datatypes,...
    'apvals',apvals,'meanapvals',meanapvals,'distname',kappadistname,'netname',nets{ndx},'exptype',exptype);
end


if dosavefig,
  if isempty(savename),
    savename = fullfile(savedir,sprintf('apoksdata_%s_%s.tex',kappadistname,exptype));
  end
  fid = fopen(savename,'w');
else
  fid = 1;
end
fprintf(fid,'distname = %s\\\\\n',kappadistname);
fprintf(fid,'kappa fit:\\\\\n');
for i = 1:npttypes,
  fprintf(fid,'  %s = %.2f\\\\\n',pttypes{i,1},kappa(i));
end
fprintf(fid,'AP averaged over OKS = %s\\\\\n\n',mat2str(meanapvals));
for datai = ndatatypes,
  for labeli = 1:nlabeltypes,
    fprintf(fid,['\\begin{tabular}{|c||',repmat('c|',[1,nnets+numel(annfns)]),'}']);
    fprintf(fid,'\\hline\n');
    fprintf(fid,'Measure - %s',labeltypes{labeli});
    for i = 1:nnets,
      fprintf(fid,' & %s',legendnames{i});
    end
    for i = 1:numel(annfns),
      fprintf(fid,' & %s',annfns{i});
    end
    fprintf(fid,'\\\\\\hline\\hline\n');
    
    fprintf(fid,'AP');
    PrintTableLine(fid,cellfun(@(x) x(1,datai,labeli),ap),nnets);
%     for ndx = 1:nnets+numel(annfns),
%       fprintf(fid,' & %.2f',ap{ndx}(1,datai,labeli));
%     end
%     fprintf(fid,'\\\\\\hline\n');
    for pti = 1:npttypes,
      fprintf(fid,'AP/%s',pttypes{pti,1});
      PrintTableLine(fid,cellfun(@(x) x(1+pti,datai,labeli),ap),nnets);
%       for ndx = 1:nnets+numel(annfns),
%         fprintf(fid,' & %.2f',ap{ndx}(1+pti,datai,labeli));
%       end
%       fprintf(fid,'\\\\\\hline\n');
    end
    for k = 1:numel(apvals),
      fprintf(fid,'AP-OKS=%d',apvals(k));
      PrintTableLine(fid,cellfun(@(x) x(k,1,datai,labeli),apk),nnets);
%       for ndx = 1:nnets+numel(annfns),
%         fprintf(fid,' & %.2f',apk{ndx}(k,1,datai,labeli));
%       end
%       fprintf(fid,'\\\\\\hline\n');
    end
    for pti = 1:npttypes,
      for k = 1:numel(apvals),
        fprintf(fid,'AP-OKS=%d/%s',apvals(k),pttypes{pti,1});
        PrintTableLine(fid,cellfun(@(x) x(k,1+pti,datai,labeli),apk),nnets);
%         for ndx = 1:nnets+numel(annfns),
%           fprintf(fid,' & %.2f',apk{ndx}(k,1+pti,datai,labeli));
%         end
%         fprintf(fid,'\\\\\\hline\n');
      end
    end
    fprintf(fid,'\\end{tabular}\n\n');
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

