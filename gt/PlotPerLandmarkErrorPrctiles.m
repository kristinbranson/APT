function [hfigs,savenames,errprctiles] = PlotPerLandmarkErrorPrctiles(varargin)

[errprctiles,gtdata,nets,legendnames,colors,colorfracprcs,exptype,...
  conddata,labeltypes,datatypes,...
  maxerr,prcs,...
  pttypes,...
  hfigs,figpos,savedir,savenames,dosavefig] = myparse(varargin,'errprctiles',{},'gtdata',[],...
  'nets',{},'legendnames',{},'colors',[],'colorfracprcs',[],'exptype','exp',...
  'conddata',[],'labeltypes',{},'datatypes',{},...
  'maxerr',[],...
  'prcs',[50 75 90 95 97],...
  'pttypes',{},...
  'hfigs',[],'figpos',[],...
  'savedir','.',...
  'savenames',{},...
  'dosavefig',false);

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
  if isempty(pttypes),
    npttypes = nlandmarks;
  end
else
  npttypes = size(errprctiles,4);
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

if isempty(colorfracprcs),
  colorfracprcs = linspace(1,.01,nprcs);
end
  

if isempty(errprctiles),
  errprctiles = nan([nprcs,nnets,npttypes,nlabeltypes,ndatatypes]);

  for ndx = 1:nnets,
    mndx = numel(gtdata.(nets{ndx}));
    cur_data = gtdata.(nets{ndx}){mndx};
    preds = cur_data.pred;
    labels = cur_data.labels;
    assert(all(size(preds)==size(labels)));
%     assert(size(preds,3)==2);
%     assert(size(labels,3)==2);
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
        
        
        for typei = 1:npttypes,
          %ptis = typei;
          ptis = pttypes{typei,2};
          dist = sqrt(sum( (preds(idx,ptis,:)-labels(idx,ptis,:)).^2,3));
          errprctiles(:,ndx,typei,labeli,datai) = prctile(dist(:),prcs);
        end
        
      end
    end
  end
end

if isempty(maxerr),
  maxerr = max(cat(1,errprctiles(:)));
end

if isempty(figpos),
  figpos = [10,10,150+(2080-150)/4*nlabeltypes/2,520/2];
end

offs = linspace(-.3,.3,nnets);
doff = offs(2)-offs(1);
dx = [-1,1,1,-1,-1]*doff/2;
dy = [0,0,1,1,0];

maxerrplus = maxerr*1.05;

for datai = 1:ndatatypes,
  
  if numel(hfigs) >= datai,
    figure(hfigs(datai));
  else
    hfigs(datai) = figure;
  end
  clf;
  set(hfigs(datai),'Position',figpos);
  hax = createsubplots(1,nlabeltypes,[[.05,.005];[.1,.005]]);
  hax = reshape(hax,[1,nlabeltypes]);
  
  for labeli = 1:nlabeltypes,
    axes(hax(labeli));
    hold on;
    
    h = gobjects(1,nnets);
    for pti = 1:npttypes,
      
      for ndx = 1:nnets,
        %mndx = numel(gtdata.(nets{ndx}));
        mndx = 2;
        offx = pti + offs(ndx);
        for prci = nprcs:-1:1,
          colorfrac = colorfracprcs(prci);
          tmp = [0,errprctiles(prci,ndx,pti,labeli,datai)];
          patch(offx+dx,tmp(1+dy),colors(ndx,:)*colorfrac + 1-colorfrac,'EdgeColor',colors(ndx,:));
        end
        if pti == 1 && labeli == 1,
          h(ndx) = patch(nan(size(dx)),nan(size(dx)),colors(ndx,:),'LineStyle','none');
        end
      end
    end
    
    set(gca,'XTick',1:npttypes,'XTickLabels',pttypes,'XtickLabelRotation',45);
    
    if labeli == 1,
      hprcs = gobjects(1,nprcs);
      for prci = 1:nprcs,
        hprcs(prci) = patch(nan(size(dx)),nan(size(dx)),[1,1,1]-colorfracprcs(prci),'EdgeColor','k');
      end
      legend([h,hprcs],[legendnames,arrayfun(@(x) sprintf('%d %%ile',x),prcs,'Uni',0)]);
    end
    if labeli == 1,
      ylabel('Error (px)');
    end
    title(labeltypes{labeli,1},'FontWeight','normal');
    set(gca,'XLim',[0,npttypes+1],'YLim',[0,maxerrplus]);
    drawnow;
  end
  set(hax(2:end),'YTickLabel',{});
  
  set(hfigs(datai),'Name',datatypes{datai,1});
  set(hfigs(datai),'Renderer','painters');
  if dosavefig,
    if numel(savenames) < datai,
      savenames{datai} = fullfile(savedir,sprintf('%s_GTTrackingError_FinalPrctiles_%s.svg',exptype,datatypes{datai,1}));
      saveas(hfigs(datai),savenames{datai},'svg');
    end
  end
end
