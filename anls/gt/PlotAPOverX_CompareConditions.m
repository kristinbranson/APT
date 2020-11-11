function [hfig,savenames] = PlotAPOverX_CompareConditions(varargin)

[gtdata,nets,legendnames,colors,exptype,...
  n_train,conddata,labeltypes,datatypes,...
  statname,hfig,figpos,savedir,savenames,dosavefig,xl,savekey,logscaley,...
  dataallonly,labelallonly,extras] = ...
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
  'dataallonly',true,...
  'labelallonly',false);

if isempty(nets),
  assert(~isempty(gtdata));
  nets = fieldnames(gtdata);
else
  ism = ismember(nets,fieldnames(gtdata));
  nets = nets(ism);
  legendnames = legendnames(ism);
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

allonly = labelallonly && dataallonly;

if allonly,
  assert('not implemented');
elseif labelallonly,
  nconditions = ndatatypes;
elseif dataallonly,
  nconditions = nlabeltypes;
else
  nconditions = ndatatypes.*nlabeltypes;
end


if isempty(legendnames),
  legendnames = nets;
end

if isempty(colors),
  colors = lines(nconditions);
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

nmodels = nan(1,nnets);
for ndx = 1:nnets,
  nmodels(ndx) = numel(gtdata.(nets{ndx}));
end
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
nc = ceil(sqrt(nnets));
nr = ceil(nnets/nc);
clf;
set(hfig,'Position',figpos);
hax = createsubplots(nr,nc,[[.05,.025];[.05,.025]]);
hax = reshape(hax,[nr,nc]);

yoff = 1.0001;
maxAP = max(AWP(:));

for ndx = 1:nnets,

  if ~isfield(gtdata,nets{ndx}),
    continue;
  end

  axes(hax(ndx));
  cla(hax(ndx));
  hold on;
  h = gobjects(1,nconditions);

  n_models_curr = numel(n_train{ndx});
  if n_train_match,
    xcurr = 1:n_models_curr;
  else
    xcurr = n_train{ndx};
  end
  
  conditioni = 0;
  conditionnames = cell(1,nconditions);
  for datai = 1:ndatatypes,

    if ndatatypes > 1 && dataallonly && ~strcmp(datatypes{datai,1},'all'),
      continue;
    end
    
    for labeli = 1:nlabeltypes,

      if nlabeltypes > 1 && labelallonly && ~strcmp(labeltypes{labeli,1},'all'),
        continue;
      end

      conditioni = conditioni + 1;
      conditionnames{conditioni} = sprintf('%s, %s',datatypes{datai,1},labeltypes{labeli,1});
      ycurr = squeeze(AWP(1:n_models_curr,ndx,datai,labeli))';
      if logscaley,
        h(conditioni) = plot(xcurr,yoff-ycurr,'.-','LineWidth',2,'Color',colors(conditioni,:),'MarkerSize',12);
      else
        h(conditioni) = plot(xcurr,ycurr,'.-','LineWidth',2,'Color',colors(conditioni,:),'MarkerSize',12);
      end

    end
      
    if n_train_match,
      set(gca,'XTick',1:numel(n_train{n_train_idx}),'XTickLabels',num2str(n_train{n_train_idx}(:)));
    end
  end

  if ndx == 1,
    legend(h(ishandle(h)),conditionnames(ishandle(h)));
  end
  title(legendnames{ndx});
  xlabel(xl);
  if n_train_match,
    set(gca,'XLim',[.5,n_models-.5]);
  end
  if logscaley,
    set(gca,'YDir','reverse','YScale','log');
    yticks = fliplr([.25,.5,.75,.9,.95,.975,.99]);
    yticks0 = yoff-yticks;
    ylim = get(gca,'YLim');    
    yticks(yticks0 < ylim(1) | yticks0 > ylim(2)) = [];
    if isempty(yticks),
      yticks0 = get(gca,'YTick');
      yticks = yoff-yticks0;
    end
    set(gca,'YTick',yoff-yticks,'YTickLabel',num2str(yticks(:)));%,'YLim',yoff-[maxAP,0]);
  else
    set(gca,'YLim',[0,1]);
  end
  drawnow;
end
set(hfig,'Renderer','painters');  
  
if dosavefig,
  assert(~isempty(savekey));
  savestr = sprintf('%s_AWP',exptype);
  if logscaley,
    savestr = [savestr,'_logy'];
  end
  if dataallonly,
    savestr = [savestr,'_labeltypes'];
  elseif labelallonly,
    savestr = [savestr,'_datatypes'];
  else
    savestr = [savestr,'_condition'];
  end
  savename = fullfile(savedir,sprintf('%s_%s.svg',savestr,savekey));
  fprintf('Saving to %s...\n',savename);
  saveas(hfig,savename,'svg');
end