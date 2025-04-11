function [hfig,savename,sortederrs] = PlotFracInliersPerLandmark_AR(varargin)

[gtdata,net,pttypes,colors,exptype,...
  conddata,labeltypes,datatypes,statname,...
  maxerr,minerr,...
  hfig,figpos,savedir,savename,dosavefig,prcs,maxprc,...
  annoterrdata,annotcolor,annoterrprc,plotannfracinliers,...
  APthresh] = myparse(varargin,'gtdata',[],...
  'net','','pttypes',{},'colors',[],'exptype','exp',...
  'conddata',[],'labeltypes',{},'datatypes',{},'statname','Worst',...
  'maxerr',[],'minerr',[],...
  'hfig',[],'figpos',[],...
  'savedir','.',...
  'savename','',...
  'dosavefig',false,...
  'prcs',[],...
  'maxprc',100,...
  'annoterrdata',[],'annotcolor',[0,0,0],'annoterrprc',[],'plotannfracinliers',true,...
  'APthresh',[]);

assert(~isempty(net));

npts = size(gtdata.(net){end}.labels,1);
nlandmarks = size(gtdata.(net){end}.labels,2);
if isempty(pttypes),
  pttypes = cell(1,nlandmarks);
  for i = 1:nlandmarks,
    pttypes{i,1} = num2str(i);
    pttypes{i,2} = i;
  end
end
npttypes = size(pttypes,1);


if isempty(conddata),
  conddata = struct;
  conddata.data_cond = ones(npts,1);
  conddata.label_cond = ones(npts,1);
end

if isempty(labeltypes),
  nlabeltypes = max(conddata.label_cond);
  labeltypes = cell(nlabeltypes,2);
  for i = 1:nlabeltypes,
    labeltypes{i,1} = num2str(i);
    labeltypes{i,2} = i;
  end
end
if isempty(datatypes),
  ndatatypes = max(conddata.data_cond);
  datatypes = cell(ndatatypes,2);
  for i = 1:ndatatypes,
    datatypes{i,1} = num2str(i);
    datatypes{i,2} = i;
  end
end
ndatatypes = size(datatypes,1);
nlabeltypes = size(labeltypes,1);

legendnames = pttypes(:,1);

if isempty(colors),
  colors = lines(npttypes);
end

sortederrs = cell([npttypes+1,nlabeltypes,ndatatypes]);
cur_data = gtdata.(net){end};

for pti = 1:npttypes,
  
  sortederrs(pti,:,:) = GetSortedErrs(cur_data,pttypes{pti,2},net,exptype,conddata,datatypes,labeltypes,'all');
  
end
% sortederrs(end,:,:) = GetSortedErrs(cur_data,1:nlandmarks,net,exptype,conddata,datatypes,labeltypes,statname);

% if ~isempty(annoterrdata),
%   if ~isfield(annoterrdata,'pred'),
%     annfns = fieldnames(annoterrdata);
%   else
%     annfns = {'annotator'};
%   end
%   annotsortederrs = cell([numel(annfns),nlabeltypes,ndatatypes]);
%   for i = 1:numel(annfns),
%     if ~isfield(annoterrdata,'pred'),
%       cur_data = annoterrdata.(annfns{i}){end};
%     else
%       cur_data = annoterrdata;
%     end
%     annotsortederrs(i,:,:) = GetSortedErrs(cur_data,annfns{i},'',cur_data,datatypes,labeltypes,statname);
%   end
%   if ~isempty(annoterrprc),
%     annoterrthresh = nan([nlabeltypes,ndatatypes]);
%     for labeli = 1:nlabeltypes,
%       for datai = 1:ndatatypes,
%         annoterrthresh(labeli,datai) = prctile( cat(1,annotsortederrs{:,labeli,datai}),annoterrprc );
%       end
%     end
%   end
%   if plotannfracinliers,
%     sortederrs = cat(1,sortederrs,annotsortederrs);
%   end
% else
%   annfns = {};
% end

ncurrs = cellfun(@numel,sortederrs);
maxn = max(ncurrs(:));

if isempty(maxerr),
  maxerr = max(cat(1,sortederrs{:}));
end
if isempty(minerr),
  minerr = min(cat(1,sortederrs{:}));
  if minerr >= maxerr,
    maxerr = max(cat(1,sortederrs{:}));
  end
end


% if isempty(figpos),
%   figpos = [10,10,2526/5,(150+1004/4*nlabeltypes)/2];
% end
if isempty(figpos),
  figpos = [10,10,720,420];
end

if isempty(hfig),
  hfig = figure;
else
  figure(hfig);
end
clf;

set(hfig,'Units','pixels','Position',figpos);
hax = axes;
% hax = createsubplots(nlabeltypes,ndatatypes,[[.1,.005];[.15,.005]]);
% hax = reshape(hax,[nlabeltypes,ndatatypes]);
  
maxprc = min(maxprc,(1-1/maxn)*100);

alllegendnames = [legendnames(:)];
% if ~isempty(annoterrdata) && plotannfracinliers,
%   alllegendnames = [alllegendnames;annfns(:)];
%   if numel(annfns) == 1,
%     grays = [0,0,0];
%   else
%     grays = linspace(0,.4,numel(annfns))'+[0,0,0];
%   end
%   colors = cat(1,colors,grays);
%   nplot = nnets + numel(annfns);
% else
%   nplot = nnets;
% end
nplot = npttypes+1;
colors = [colors;[0,0,0]];

for datai = ndatatypes:-1:1,
  
  for labeli = 1:nlabeltypes,
    
    axes(hax(labeli,datai));
    cla(hax(labeli,datai));
    
    hold on;
    h = gobjects(1,nplot);
    
%     if ~isempty(annoterrdata) && ~isempty(annoterrprc),
%       plot(annoterrthresh(labeli,datai)+[0,0],[eps,100],':','Color',annotcolor,'LineWidth',2);
%     end
    
    for ndx = 1:nplot
      ncurr = numel(sortederrs{ndx,labeli,datai});
      if ncurr == 0,
        continue;
      end
      h(ndx) = plot(sortederrs{ndx,labeli,datai},max(eps,100-(1:ncurr)/ncurr*100),'-','LineWidth',1,'Color',colors(ndx,:));
    end
    if labeli == nlabeltypes && datai == ndatatypes,
      legend(h(ishandle(h)),alllegendnames(ishandle(h)),'Location','East');
    end
    if labeli == nlabeltypes,
      xlabel('Error threshold (px)');
    end
    if datai == 1,
      ylabel('Precision');
    end
    if labeli == 1,
%       title(sprintf('%s',datatypes{datai,1}),'FontWeight','normal');
    end
    if isempty(prcs),
      prcs = get(gca,'YTick');
    end
    prcs = prcs(:);
    set(gca,'YTick',100-flipud(prcs),'YTickLabel',num2str(flipud(prcs(:))));
    set(gca,'XLim',[minerr,maxerr],'YLim',[100-maxprc,100]);%,'XTick',10:10:maxerr);
    set(gca,'XScale','linear','YScale','log','YDir','reverse');
    if ~isempty(APthresh),
      set(gca,'XTick',APthresh,'XTickLabel',{},'XGrid','on');
    end

    drawnow;
  end
end
linkaxes(hax);
set(hax(:,2:end),'YTickLabel',{});
set(hax(1:end-1,:),'XTickLabel',{});

set(hfig,'Name','Precision, all landmark types');
% set(hfig,'Renderer','painters');
if dosavefig,
  if isempty(savename),
    savename = fullfile(savedir,sprintf('%s_GTFracInliers_%s_GroupedKeypoints',exptype,statname));
  end
%   saveas(hfig,savename,'svg');
exportgraphics(hfig,[savename,'.png'])
exportgraphics(hfig,[savename,'.pdf'],'ContentType','vector')
end


function sortederrs = GetSortedErrs(cur_data,ptidx,net,exptype,conddata,datatypes,labeltypes,statname)

isshexp = startsWith(exptype,'SH');

preds = cur_data.pred(:,ptidx,:);
labels = cur_data.labels(:,ptidx,:);
assert(all(size(preds)==size(labels)));
% assert(size(preds,3)==2);
% assert(size(labels,3)==2);
iscpr = contains(net,'cpr');

ndatatypes = size(datatypes,1);
nlabeltypes = size(labeltypes,1);

sortederrs = cell(nlabeltypes,ndatatypes);
for datai = 1:ndatatypes,
  dtis = datatypes{datai,2};
  for labeli = 1:nlabeltypes,
    ltis = labeltypes{labeli,2};
    idx = ismember(conddata.data_cond,dtis) & ismember(conddata.label_cond,ltis);
    if iscpr && isshexp,
      % special case for SH/cpr whose computed GT output only has
      % 1149 rows instead of 1150 cause dumb
      idx(4) = [];
    end
    dist = sqrt(sum( (preds(idx,:,:)-labels(idx,:,:)).^2,3));
    switch lower(statname),
      case 'worst',
        distcurr = max(dist,[],2);
      case 'median',
        distcurr = median(dist,2);
      case 'best'
        distcurr = min(dist,[],2);
      case 'all'
        distcurr = dist(:);
        case 'mean'
            distcurr = mean(dist,2);
        
    end
    sortederrs{labeli,datai} = sort(distcurr);
  end
end
