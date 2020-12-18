function [hfigs,savenames,errprctiles] = PlotWorstLandmarkErrorOverX(varargin)

[errprctiles,gtdata,nets,legendnames,colors,colorfracprcs,exptype,...
  n_train,...
  conddata,labeltypes,datatypes,...
  maxerr,prcs,...
  statname,...
  hfigs,figpos,savedir,savenames,dosavefig,xl,savekey] = myparse(varargin,'errprctiles',{},'gtdata',[],...
  'nets',{},'legendnames',{},'colors',[],'colorfracprcs',[],'exptype','exp',...
  'n_train',[],...
  'conddata',[],'labeltypes',{},'datatypes',{},...
  'maxerr',[],...
  'prcs',[50 75 90 95 97],...
  'statname','Worst',...
  'hfigs',[],'figpos',[],...
  'savedir','.',...
  'savenames',{},...
  'dosavefig',false,...
  'x','',...
  'savekey','');

prcs = sort(prcs);

isshexp = startsWith(exptype,'SH');

if isempty(nets),
  assert(~isempty(gtdata));
  nets = fieldnames(gtdata);
end
nnets = numel(nets);
nprcs = numel(prcs);

if isempty(errprctiles),
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
  
else
  n_models = size(errprctiles,3)+1; % first model is ignored
end

if isempty(conddata) || ~exist('ndatatypes','var')
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

if isempty(colorfracprcs),
  colorfracprcs = linspace(1,.01,nprcs);
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

if isempty(errprctiles),
  errprctiles = nan([nprcs,nnets,n_models-1,nlabeltypes,ndatatypes]);

  for ndx = 1:nnets,

    if ~isfield(gtdata,nets{ndx}),
      continue;
    end
    
    for mndx = 2:n_models
      if numel(gtdata.(nets{ndx})) < mndx,
        break;
      end
      
      cur_data = gtdata.(nets{ndx}){mndx};
      preds = cur_data.pred;
      labels = cur_data.labels;
      assert(all(size(preds)==size(labels)));
%       assert(size(preds,3)==2);
%       assert(size(labels,3)==2);
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
          errprctiles(:,ndx,mndx-1,labeli,datai) = prctile(distcurr,prcs);
        end
      end
    end
  end
end

if isempty(maxerr),
  maxerr = max(cat(1,errprctiles(:)));
end
maxerrplus = maxerr*1.05;

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
for datai = 1:ndatatypes,
  
  if numel(hfigs) < datai,
    hfigs(datai) = figure;
  else
    figure(hfigs(datai));
  end
  clf;
  set(hfigs(datai),'Position',figpos);
  hax = createsubplots(nlabeltypes,nprcs,[[.025,.005];[.1,.005]]);
  hax = reshape(hax,[nlabeltypes,nprcs]);
  
  for labeli = 1:nlabeltypes,
    
    for prci = 1:nprcs,
      
      axes(hax(labeli,prci));
      cla(hax(labeli,prci));
      
      hold on;
      
      h = gobjects(1,nnets);
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
        h(ndx) = plot(xcurr,squeeze(errprctiles(prci,ndx,1:n_models_curr,labeli,datai))','.-','LineWidth',2,'Color',colors(ndx,:),'MarkerSize',12);

      end
      
      if n_train_match,
        set(gca,'XTick',1:numel(n_train{n_train_idx}),'XTickLabels',num2str(n_train{n_train_idx}(:)));
        set(gca,'XLim',[0,n_models]);
      end
      
      if labeli == 1 && prci == 1,
        legend(h(ishandle(h)),legendnames(ishandle(h)));
      end
      if labeli == nlabeltypes && ~isempty(xl),
        xlabel(xl);
      end
      if prci == 1,
        ylabel(labeltypes{labeli,1});
      end
      if labeli == 1,
        title(sprintf('%s landmark, %dth %%ile',statname,prcs(prci)),'FontWeight','normal');
      end
      set(gca,'YLim',[0,maxerrplus]);
      drawnow;
    end
    set(hax(:,2:end),'YTickLabel',{});
    set(hax(1:end-1,:),'XTickLabel',{});
    
  end
  set(hfigs(datai),'Name',sprintf('%s landmark, %s',statname,datatypes{datai,1}));
  set(hfigs(datai),'Renderer','painters');
  
  
  if dosavefig,
    assert(~isempty(savekey));
    if numel(savenames) < datai,
      savenames{datai} = fullfile(savedir,sprintf('%s_GTTrackingError_%s_%sLandmark_%s.svg',exptype,savekey,statname,datatypes{datai,1}));
    end
    saveas(hfigs(datai),savenames{datai},'svg');
  end
  %break;
end
