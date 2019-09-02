function [hfig,savename,sortederrs] = PlotFracInliers(varargin)

[sortederrs,gtdata,nets,legendnames,colors,exptype,...
  conddata,labeltypes,datatypes,statname,...
  maxerr,minerr,...
  hfig,figpos,savedir,savename,dosavefig,prcs,maxprc,...
  annoterrdata,annotcolor,annoterrprc,plotannfracinliers,APthresh,xscale,...
  anglefn,convert2deg] = myparse(varargin,'sortederrs',{},'gtdata',[],...
  'nets',{},'legendnames',{},'colors',[],'exptype','exp',...
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
  'xscale','linear',...
  'anglefn','',...
  'convert2deg',false);

if isempty(nets),
  assert(~isempty(gtdata));
  nets = fieldnames(gtdata);
end
nnets = numel(nets);

if isempty(sortederrs),
  assert(~isempty(gtdata));
  if isempty(anglefn),
    npts = size(gtdata.(nets{1}){end}.labels,1);
  else
    npts = size(gtdata.(nets{1}){end}.(anglefn).labels,1);
  end
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
    sortederrs(ndx,:,:) = GetSortedErrs(cur_data,nets{ndx},exptype,conddata,datatypes,labeltypes,statname,anglefn,convert2deg);

  end
end

if ~isempty(annoterrdata),
  if ~isfield(annoterrdata,'pred'),
    annfns = fieldnames(annoterrdata);
  else
    annfns = {'annotator'};
  end
  annotsortederrs = cell([numel(annfns),nlabeltypes,ndatatypes]);
  for i = 1:numel(annfns),
    if ~isfield(annoterrdata,'pred'),
      cur_data = annoterrdata.(annfns{i}){end};
    else
      cur_data = annoterrdata;
    end
    annotsortederrs(i,:,:) = GetSortedErrs(cur_data,annfns{i},'',cur_data,datatypes,labeltypes,statname,anglefn,convert2deg);
  end
  if ~isempty(annoterrprc),
    annoterrthresh = nan([nlabeltypes,ndatatypes]);
    for labeli = 1:nlabeltypes,
      for datai = 1:ndatatypes,
        annoterrthresh(labeli,datai) = prctile( cat(1,annotsortederrs{:,labeli,datai}),annoterrprc );
      end
    end
  end
  if plotannfracinliers,
    sortederrs = cat(1,sortederrs,annotsortederrs);
  end
else
  annfns = {};
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

alllegendnames = legendnames(:);
if ~isempty(annoterrdata) && plotannfracinliers,
  alllegendnames = [alllegendnames;annfns(:)];
  if numel(annfns) == 1,
    grays = [0,0,0];
  else
    grays = linspace(0,.4,numel(annfns))'+[0,0,0];
  end
  colors = cat(1,colors,grays);
  nplot = nnets + numel(annfns);
else
  nplot = nnets;
end

for datai = ndatatypes:-1:1,
  
  for labeli = 1:nlabeltypes,
    
    axes(hax(labeli,datai));
    cla(hax(labeli,datai));
    
    hold on;
    h = gobjects(1,nplot);
    
    if ~isempty(annoterrdata) && ~isempty(annoterrprc),
      plot(annoterrthresh(labeli,datai)+[0,0],[eps,100],':','Color',annotcolor,'LineWidth',2);
    end
    
    for ndx = 1:nplot
      ncurr = numel(sortederrs{ndx,labeli,datai});
      if ncurr == 0,
        h(ndx) = plot(nan(1,2),nan(1,2),'-','LineWidth',2,'Color',colors(ndx,:));
        continue;
      end
      h(ndx) = plot(sortederrs{ndx,labeli,datai},max(eps,100-(1:ncurr)/ncurr*100),'-','LineWidth',2,'Color',colors(ndx,:));
    end
    if labeli == nlabeltypes,
      if isempty(anglefn),
        xl = sprintf('%s landmark error',statname);
      else
        if convert2deg,
          xl = sprintf('%s error (deg)',anglefn);
        else
          xl = sprintf('%s error (rad)',anglefn);
        end
      end
      xlabel(xl,'Interpreter','none');
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
    %xstep = 10^floor(log10(maxerr));
    set(gca,'XLim',[minerr,maxerr],'YLim',[100-maxprc,100]);%,'XTick',10:10:maxerr);
    set(gca,'XScale',xscale,'YScale','log','YDir','reverse');
    if ~isempty(APthresh),
      for threshi = 1:numel(APthresh),
        plot(APthresh(threshi)+[0,0],[100-maxprc,100],'-','Color',[.7,.7,.7]);
      end
      %       for threshi = 1:numel(APthresh),
      %         plot(APthresh(threshi)+[0,0],[eps,100],':','Color',annotcolor,'LineWidth',2);
      %       end
    end
    if labeli == 1 && datai == 1,
      legend(h,alllegendnames);
    end

    drawnow;
  end
end
linkaxes(hax);
set(hax(:,2:end),'YTickLabel',{});
set(hax(1:end-1,:),'XTickLabel',{});

if isempty(anglefn),
  set(hfig,'Name',sprintf('%s landmark',statname));
else
  set(hfig,'Name',anglefn);
end
set(hfig,'Renderer','painters');
if dosavefig,
  if isempty(savename),
    if isempty(anglefn),
      plottype = sprintf('%sLandmark',statname);
    else
      plottype = anglefn;
    end
    if strcmpi(xscale,'linear'),
      savename = fullfile(savedir,sprintf('%s_GTFracInliers_%s.svg',exptype,plottype));
    else
      savename = fullfile(savedir,sprintf('%s_GTFracInliers_logx_%s.svg',exptype,plottype));
    end
  end
  saveas(hfig,savename,'svg');
end


function sortederrs = GetSortedErrs(cur_data,net,exptype,conddata,datatypes,labeltypes,statname,anglefn,convert2deg)

isshexp = startsWith(exptype,'SH');

if isempty(anglefn),
  preds = cur_data.pred;
  labels = cur_data.labels;
  isangle = false;
else
  preds = cur_data.(anglefn).pred;
  labels = cur_data.(anglefn).labels;
  isangle = true;
end
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
    if isangle,
      dist = abs(modrange(preds(idx,:,:)-labels(idx,:,:),-pi,pi));
      if convert2deg,
        dist = dist*180/pi;
      end
    else
      dist = sqrt(sum( (preds(idx,:,:)-labels(idx,:,:)).^2,3));
    end
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
