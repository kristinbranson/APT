function [hfig,savename,sortederrs] = PlotFracInliersPerGroup_AR(varargin)

[gtdata,net,pttypes,colors,exptype,...
  conddata,labeltypes,datatypes,statname,...
  maxerr,minerr,...
  hfig,figpos,savedir,savename,dosavefig,prcs,maxprc,...
  annoterrdata,annotcolor,annoterrprc,plotannfracinliers,...
  APthresh,someallonly,labelallonly,dataallonly,lw,linestyles] = myparse(varargin,'gtdata',[],...
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
  'APthresh',[],...
  'someallonly',false,...
  'labelallonly',false,...
  'dataallonly',false,...
  'LineWidth',1,...
  'LineStyles',{'-'});

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

assert((double(dataallonly) + double(someallonly) + double(labelallonly)) <= 1);

plottype = '';
if someallonly,
  nplot = ndatatypes + nlabeltypes - 1;
  plottype = '_someall';
elseif dataallonly,
  nplot = nlabeltypes;
  plottype = '_labels';
elseif labelallonly,
  nplot = ndatatypes;
  plottype = '_data';
else
  nplot = ndatatypes*nlabeltypes;
end

if isempty(colors),
  colors = lines(nplot);
end

cur_data = gtdata.(net){end};

sortederrs = GetSortedErrs(cur_data,1:nlandmarks,net,exptype,conddata,datatypes,labeltypes,statname);

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
hax = gca;
%hax = reshape(hax,[nlabeltypes,ndatatypes]);
  
maxprc = min(maxprc,(1-1/maxn)*100);

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
%nplot = npttypes+1;

hold on;
h = gobjects(nplot,1);
ploti = 1;
legendnames = cell(nplot,1);

for datai = 1:ndatatypes,
  
  for labeli = 1:nlabeltypes,

    if someallonly
      if ~strcmp(labeltypes{labeli,1},'all') && ~strcmp(datatypes{datai,1},'all'),
        continue;
      end
    elseif dataallonly
        legendnames{ploti} = sprintf('%s',labeltypes{labeli});
      if ~strcmp(datatypes{datai,1},'all'),
        continue;
      end
    elseif labelallonly,
        legendnames{ploti} = sprintf('%s',datatypes{datai});
      if ~strcmp(labeltypes{labeli,1},'all'),
        continue;
      end
    end

%     if ~isempty(annoterrdata) && ~isempty(annoterrprc),
%       plot(annoterrthresh(labeli,datai)+[0,0],[eps,100],':','Color',annotcolor,'LineWidth',lw);
%     end
    
    ncurr = numel(sortederrs{labeli,datai});
    ls = linestyles{min(ploti,numel(linestyles))};
    if ncurr == 0,
      h(ploti) = plot(nan(1,2),nan(1,2),ls,'LineWidth',lw,'Color',colors(ploti,:));
      continue;
    end
    h(ploti) = plot(sortederrs{labeli,datai},max(eps,100-(1:ncurr)/ncurr*100),ls,'LineWidth',lw,'Color',colors(ploti,:));
    
    ploti = ploti+1;
  end
end
xlabel('Error Threshold (px)');
ylabel('Precision');
if isempty(prcs),
  prcs = 100-get(gca,'YTick');
end
set(gca,'YTick',100-flipud(prcs(:)),'YTickLabel',num2str(flipud(prcs(:))));
xstep = 10^(floor(log10(maxerr)));
set(gca,'XLim',[minerr,maxerr],'YLim',[100-maxprc,100],'XTick',xstep:xstep:maxerr);
set(gca,'XScale','linear','YScale','log','YDir','reverse');

if ~isempty(APthresh),
  for threshi = 1:numel(APthresh),
    plot(APthresh(threshi)+[0,0],[100-maxprc,100],'-','Color',[.7,.7,.7]);
  end
end
legend(h(ishandle(h)),legendnames(ishandle(h)),'Location','East');

set(hfig,'Name','Precision, all groups');
% set(hfig,'Renderer','painters');
if dosavefig,
  if isempty(savename),
    savename = fullfile(savedir,sprintf('%s_GTFracInliers_%s_AllGroups%s_%s',exptype,net,plottype,statname));
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
