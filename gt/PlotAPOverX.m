function [hfig,savenames] = PlotAPOverX(varargin)

[gtdata,nets,legendnames,colors,exptype,...
  n_train,conddata,labeltypes,datatypes,...
  statname,hfig,figpos,savedir,savenames,dosavefig,xl,savekey,logscaley,annfn,lw,extras] = ...
  myparse_nocheck(varargin,'gtdata',[],...
  'nets',{},'legendnames',{},'colors',[],'exptype','exp',...
  'n_train',[],...
  'conddata',[],'labeltypes',{},'datatypes',{},...
  'statname','Worst',...
  'hfigs',[],'figpos',[],...
  'savedir','.',...
  'savenames',{},...
  'dosavefig',false,...
  'x','',...
  'savekey','',...
  'logscaley',false,...
  'annfn','intra',...
  'LineWidth',2);

if isempty(nets),
  assert(~isempty(gtdata));
  nets = fieldnames(gtdata);
else
  ism = ismember(nets,fieldnames(gtdata));
  nets = nets(ism);
  legendnames = legendnames(ism);
  if ~isempty(colors),
    colors = colors(ism,:);
  end
end



nnets = numel(nets);

assert(~isempty(gtdata));
ndatapts = size(gtdata.(nets{1}){end}.labels,1);
if isempty(conddata),
  conddata = struct;
  conddata.data_cond = ones(ndatapts,1);
  conddata.label_cond = ones(ndatapts,1);
end
nlandmarks = size(gtdata.(nets{1}){end}.labels,2);
n_models = 0;
for ndx = 1:nnets,
  if ~isfield(gtdata,nets{ndx}),
    continue;
  end
  n_models = max(n_models,numel(gtdata.(nets{ndx})));
end

if isempty(conddata),
  ndatatypes = 1;
  nlabeltypes = 1;
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

if isempty(n_train),
  assert(~isempty(gtdata));
  n_train = cell(1,nnets);
  for i = 1:nnets,
    if ~isfield(gtdata,nets{i}),
      continue;
    end
    n_train{i} = cellfun(@(x) x.model_timestamp,gtdata.(nets{i})(2:end));
  end
end
% whether we have the same x-axes for each net
n_train_match = true;
n_train_idx = argmax(cellfun(@numel,n_train));
for i = 1:numel(nets),  
  ncurr = numel(n_train{i});
  if ~all(n_train{i}(1:ncurr)==n_train{n_train_idx}(1:ncurr)),
    n_train_match = false;
    break;
  end
end

if strcmpi(savekey,'TrainTime'),
  for ndx = 1:nnets,
    if ~isfield(gtdata,nets{ndx}),
      continue;
    end
    n_train{ndx} = (n_train{ndx}-n_train{ndx}(1))/3600;
  end
end

% compute ap

nmodels = structfun(@numel,gtdata);
maxnmodels = max(nmodels);

AWP = nan([maxnmodels,nnets,ndatatypes,nlabeltypes]);

for modeli = 1:maxnmodels,
  [tbls] = ComputePixelPrecisionTable(gtdata,...
    'nets',nets,'exptype',exptype,'savename','','dosavefig',false,...
    'conddata',conddata','labeltypes',labeltypes,'datatypes',datatypes,...
    extras{:},'modeli',modeli);
  idx = nmodels <= maxnmodels;
  for datai = 1:ndatatypes,
    for labeli = 1:nlabeltypes,
      AWP(modeli,idx,datai,labeli) = tbls{datai,labeli}.AWP(idx);
    end
  end
end

AWP_ann = nan(ndatatypes,nlabeltypes);
idx = find(strcmp(tbls{end,end}.Row,annfn));
if ~isempty(idx),
  for datai = 1:ndatatypes,
    for labeli = 1:nlabeltypes,
      AWP_ann(datai,labeli) = tbls{datai,labeli}.AWP(idx);
    end
  end
  legendnames{end+1} = 'Annotator';
end

if isempty(figpos),
  figpos = [10,10,2526/2,(150+1004/4*nlabeltypes)/2];
end

% if n_train_match,
%   offs = linspace(-.3,.3,nnets);
%   doff = offs(2)-offs(1);
%   dx = [-1,1,1,-1,-1]*doff/2;
% else
%   max_train_time = max([n_train{:}]);
%   patchr = max_train_time /100;
%   dx = [-1,1,1,-1,-1]*patchr;
% end
% 
% dy = [0,0,1,1,0];
hfig = figure;
clf;
set(hfig,'Position',figpos);
hax = createsubplots(ndatatypes,nlabeltypes,[[.05,.025];[.05,.025]]);
hax = reshape(hax,[ndatatypes,nlabeltypes]);

yoff = 1.0001;
maxAP = max(max(AWP(:)),max(AWP_ann(:)));
if n_train_match,
  xlim = [.5,n_models-.5];
else
  xlim = [min(cellfun(@min,n_train)),max(cellfun(@max,n_train))];
  dx = diff(xlim);
  xlim = xlim+[1,-1]*dx*.025;
end

for datai = 1:ndatatypes,
  
  for labeli = 1:nlabeltypes,
          
    axes(hax(datai,labeli));
    cla(hax(datai,labeli));
    
    hold on;
      
    h = gobjects(1,nnets+1);
    if ~isnan(AWP_ann(datai,labeli)),
      if logscaley,
        ycurr = yoff-AWP_ann(datai,labeli);
      else
        ycurr = AWP_ann(datai,labeli);
      end
      plot(xlim,[0,0]+ycurr,'k:','LineWidth',lw);
    end
    for ndx = 1:nnets,
      
      if ~isfield(gtdata,nets{ndx}),
        continue;
      end
      
      n_models_curr = numel(n_train{ndx});
      if n_train_match,
        xcurr = 1:n_models_curr;
      else
        xcurr = n_train{ndx};
      end
      ycurr = squeeze(AWP(1:n_models_curr,ndx,datai,labeli))';
      if logscaley,
        h(ndx) = plot(xcurr,yoff-ycurr,'.-','LineWidth',lw,'Color',colors(ndx,:),'MarkerSize',12);
      else
        h(ndx) = plot(xcurr,ycurr,'.-','LineWidth',lw,'Color',colors(ndx,:),'MarkerSize',12);
      end

    end
    
      
    if n_train_match,
      set(gca,'XTick',1:numel(n_train{n_train_idx}),'XTickLabels',num2str(n_train{n_train_idx}(:)));
    end
    
    if labeli == 1 && datai == 1,
      legend(h(ishandle(h)),legendnames(ishandle(h)));
    end
    title(datatypes{datai,1});
    xlabel(xl);
    ylabel(labeltypes{labeli,1});
    set(gca,'XLim',xlim);
    if logscaley,
      yticks = fliplr([.25,.5,.75,.9,.95,.975,.99]);
      yticks(yticks > maxAP) = [];
      set(gca,'YDir','reverse','YScale','log','YTick',yoff-yticks,'YTickLabel',num2str(yticks(:)));%,'YLim',yoff-[maxAP,0]);
    else
      set(gca,'YLim',[0,1]);
    end
    drawnow;
  end
end
set(hfig,'Renderer','painters');
  
  
if dosavefig,
  assert(~isempty(savekey));
  if logscaley,
    savename = fullfile(savedir,sprintf('%s_AWP_logy_%s.svg',exptype,savekey));
  else
    savename = fullfile(savedir,sprintf('%s_AWP_%s.svg',exptype,savekey));
  end
  saveas(hfig,savename,'svg');
end