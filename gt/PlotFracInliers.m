function [hfig,savename,sortederrs] = PlotFracInliers(varargin)

[sortederrs,gtdata,nets,legendnames,colors,exptype,...
  conddata,labeltypes,datatypes,statname,...
  maxerr,minerr,...
  hfig,figpos,savedir,savename,dosavefig,prcs,maxprc,...
  annoterrdata,annotcolor,annoterrprc] = myparse(varargin,'sortederrs',{},'gtdata',[],...
  'nets',{},'legendnames',{},'colors',[],'exptype','exp',...
  'conddata',[],'labeltypes',{},'datatypes',{},'statname','Worst',...
  'maxerr',[],'minerr',[],...
  'hfig',[],'figpos',[],...
  'savedir','.',...
  'savename','',...
  'dosavefig',false,...
  'prcs',[],...
  'maxprc',100,...
  'annoterrdata',[],'annotcolor',[0,0,0],'annoterrprc',99);

if isempty(nets),
  assert(~isempty(gtdata));
  nets = fieldnames(gtdata);
end
nnets = numel(nets);

if isempty(sortederrs),
  assert(~isempty(gtdata));
  npts = size(gtdata.(nets{1}){end}.labels,1);
else
  npts = size(sortederrs{1},1);
end
  
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

if isempty(legendnames),
  legendnames = nets;
end

if isempty(colors),
  colors = lines(nnets);
end

if isempty(sortederrs),
  sortederrs = cell([nnets,nlabeltypes,ndatatypes]);

  for ndx = 1:nnets,
    
    cur_data = gtdata.(nets{ndx}){end};
    sortederrs(ndx,:,:) = GetSortedErrs(cur_data,nets{ndx},exptype,conddata,datatypes,labeltypes,statname);

  end
end

if ~isempty(annoterrdata),
  annotsortederrs = GetSortedErrs(annoterrdata,'annotator','',annoterrdata,datatypes,labeltypes,statname);
  annoterrthresh = cellfun(@(x) prctile(x,annoterrprc),annotsortederrs);
end

ncurrs = cellfun(@numel,sortederrs);
maxn = max(ncurrs(:));

if isempty(maxerr),
  maxerr = max(cat(1,sortederrs{:}));
end
if isempty(minerr),
  minerr = min(cat(1,sortederrs{:}));
end

if isempty(figpos),
  figpos = [10,10,2526/5,(150+1004/4*nlabeltypes)/2];
end


if isempty(hfig),
  hfig = figure;
else
  figure(hfig);
end
clf;

set(hfig,'Units','pixels','Position',figpos);
hax = createsubplots(nlabeltypes,ndatatypes,[[.025,.005];[.1,.005]]);
hax = reshape(hax,[nlabeltypes,ndatatypes]);
  
maxprc = min(maxprc,(1-1/maxn)*100);

for datai = ndatatypes:-1:1,
  
  for labeli = 1:nlabeltypes,
    
    axes(hax(labeli,datai));
    cla(hax(labeli,datai));
    
    hold on;
    h = gobjects(1,nnets);
    
    if ~isempty(annoterrdata),
      plot(annoterrthresh(labeli,datai)+[0,0],[eps,100],':','Color',annotcolor,'LineWidth',2);
    end
    
    for ndx = 1:nnets,
      ncurr = numel(sortederrs{ndx,labeli,datai});
      if ncurr == 0,
        continue;
      end
      h(ndx) = plot(sortederrs{ndx,labeli,datai},max(eps,100-(1:ncurr)/ncurr*100),'-','LineWidth',2,'Color',colors(ndx,:));
    end
    if labeli == nlabeltypes && datai == ndatatypes,
      legend(h,legendnames);
    end
    if labeli == nlabeltypes,
      xlabel(sprintf('%s landmark error',statname));
    end
    if datai == 1,
      ylabel(labeltypes{labeli,1});
    end
    if labeli == 1,
      title(sprintf('%s',datatypes{datai,1}),'FontWeight','normal');
    end
    if isempty(prcs),
      prcs = 100-get(gca,'YTick');
    end
    set(gca,'YTick',100-flipud(prcs(:)),'YTickLabel',num2str(flipud(prcs(:))));
    set(gca,'XLim',[minerr,maxerr],'YLim',[100-maxprc,100],'XTick',10:10:maxerr);
    set(gca,'XScale','log','YScale','log','YDir','reverse');
    drawnow;
  end
end
linkaxes(hax);
set(hax(:,2:end),'YTickLabel',{});
set(hax(1:end-1,:),'XTickLabel',{});

set(hfig,'Name',sprintf('%s landmark',statname));
set(hfig,'Renderer','painters');
if dosavefig,
  if isempty(savename),
    savename = fullfile(savedir,sprintf('%s_GTFracInliers_%sLandmark.svg',exptype,statname));
  end
  saveas(hfig,savename,'svg');
end


function sortederrs = GetSortedErrs(cur_data,net,exptype,conddata,datatypes,labeltypes,statname)

isshexp = startsWith(exptype,'SH');

preds = cur_data.pred;
labels = cur_data.labels;
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
    end
    sortederrs{labeli,datai} = sort(distcurr);
  end
end
