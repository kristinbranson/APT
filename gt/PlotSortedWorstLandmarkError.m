function [hfig,savename,sortederrs] = PlotSortedWorstLandmarkError(varargin)

[sortederrs,gtdata,nets,legendnames,colors,exptype,...
  conddata,labeltypes,datatypes,statname,...
  minerr,maxerr,...
  hfig,figpos,savedir,savename,dosavefig] = myparse(varargin,'sortederrs',[],'gtdata',[],...
  'nets',{},'legendnames',{},'colors',[],'exptype','exp',...
  'conddata',[],'labeltypes',{},'datatypes',{},'statname','Worst',...
  'minerr',[],'maxerr',[],...
  'hfig',[],'figpos',[],...
  'savedir','.',...
  'savename','',...
  'dosavefig',false);

isshexp = ismember(exptype,{'SHView0','SHView1'});

if isempty(nets),
  assert(~isempty(gtdata));
  nets = fieldnames(gtdata);
end
nnets = numel(nets);

if isempty(sortederrs),
  assert(~isempty(gtdata));
  npts = size(gtdata.(nets{1}){end}.labels,1);
else
  npts = size(sortederrs,1);
end
  
if isempty(conddata),
  conddata = struct;
  conddata.data_cond = ones(npts,1);
  conddata.label_cond = ones(npts,1);
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

if isempty(sortederrs),
  sortederrs = cell([nnets,nlabeltypes,ndatatypes]);

  for ndx = 1:nnets,
    mndx = numel(gtdata.(nets{ndx}));
    cur_data = gtdata.(nets{ndx}){mndx};
    preds = cur_data.pred;
    labels = cur_data.labels;
    assert(size(preds,3)==2);
    assert(size(labels,3)==2);
    iscpr = ~isempty(strfind(nets{ndx},'cpr'));

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
        sortederrs{ndx,labeli,datai} = sort(distcurr);
      end
    end
  end
end

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
  
for datai = ndatatypes:-1:1,
  
  for labeli = 1:nlabeltypes,
    
    axes(hax(labeli,datai));
    cla(hax(labeli,datai));
    
    hold on;
    h = gobjects(1,nnets);
    for ndx = 1:nnets,
      ncurr = numel(sortederrs{ndx,labeli,datai});
      if ncurr == 0,
        continue;
      end
      h(ndx) = plot((1:ncurr)/ncurr*100,squeeze(sortederrs{ndx,labeli,datai})','-','LineWidth',2,'Color',colors(ndx,:));
    end
    if labeli == nlabeltypes && datai == ndatatypes,
      legend(h,legendnames);
    end
    if labeli == nlabeltypes,
      xlabel('Error percentile');
    end
    if datai == 1,
      ylabel(labeltypes{labeli,1});
    end
    if labeli == 1,
      title(sprintf('%s, %s landmark',datatypes{datai,1},lower(statname)),'FontWeight','normal');
    end
    set(gca,'XLim',[0,100],'YLim',[minerr,maxerr]);
    set(gca,'YScale','log');
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
    savename = fullfile(savedir,sprintf('%s_ErrPercentiles_%sLandmark.svg',exptype,statname));
  end
  saveas(hfig,savename,'svg');
end
