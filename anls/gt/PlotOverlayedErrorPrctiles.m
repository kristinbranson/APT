function [hfigs,savenames,errprctilespts] = PlotOverlayedErrorPrctiles(varargin)

[freezeInfo,lpos,...
  errprctilespts,gtdata,nets,legendnames,prccolors,exptype,...
  conddata,labeltypes,datatypes,...
  prcs,...
  hfigs,figpos,savedir,savenames,dosavefig] = myparse(varargin,...
  'freezeInfo',[],'lpos',[],...
  'errprctilespts',[],'gtdata',[],...
  'nets',{},'legendnames',{},'prccolors',[],'exptype','exp',...
  'conddata',[],'labeltypes',{},'datatypes',{},...
  'prcs',[50 75 90 95 97],...
  'hfigs',[],'figpos',[],...
  'savedir','.',...
  'savenames',{},...
  'dosavefig',false);

assert(~isempty(freezeInfo));
assert(~isempty(lpos));
isshexp = startsWith(exptype,'SH');

if isempty(nets),
  assert(~isempty(gtdata));
  nets = fieldnames(gtdata);
end
nnets = numel(nets);
nprcs = numel(prcs);

if isempty(errprctilespts),
  assert(~isempty(gtdata));
  ndatapts = size(gtdata.(nets{1}){end}.labels,1);
  if isempty(conddata),
    conddata = struct;
    conddata.data_cond = ones(ndatapts,1);
    conddata.label_cond = ones(ndatapts,1);
  end
  nlandmarks = size(gtdata.(nets{1}){end}.labels,2);
else
  nlandmarks = size(errprctilespts,4);
end

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

if isempty(prccolors),
  prccolors = flipud(jet(100));
  prccolors = prccolors(round(linspace(1,100,nprcs)),:);
end

if isempty(errprctilespts),
  errprctilespts = nan([nprcs,nnets,nlandmarks,nlabeltypes,ndatatypes]);

  for ndx = 1:nnets,
    if ~isfield(gtdata,nets{ndx}),
      continue;
    end
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
        for pti=1:nlandmarks,
          dist = sqrt(sum( (preds(idx,pti,:)-labels(idx,pti,:)).^2,3));
          errprctilespts(:,ndx,pti,labeli,datai) = prctile(dist(:),prcs);
        end        
      end
    end
  end
end

if isempty(figpos),
  figpos = [10,10,1580,150+830/4*nlabeltypes];
end

for datai = 1:ndatatypes,
  
  if numel(hfigs) >= datai,
    figure(hfigs(datai));
  else
    hfigs(datai) = figure;
  end
  
  clf;
  set(hfigs(datai),'Position',figpos,'Renderer','painters','Name',datatypes{datai});
  
  hax = createsubplots(nlabeltypes,nnets,[[.025,.005];[.05,.005]]);
  hax = reshape(hax,[nlabeltypes,nnets]);
  
  for labeli = 1:nlabeltypes,
    for ndx = 1:nnets,
      axes(hax(labeli,ndx));
      imagesc(freezeInfo.xdata,freezeInfo.ydata,freezeInfo.im);
      if isfield(freezeInfo,'clim'),
        set(hax(labeli,ndx),'CLim',freezeInfo.clim);
      end
      axis image;
      colormap gray;
      box off;
      
      hold on;
      
      axcProps = freezeInfo.axes_curr;
      axfns = fieldnames(axcProps);
      %axfns = setdiff(fieldnames(axcProps),{'XLim','YLim'});
      for propi=1:numel(axfns),
        prop = axfns{propi};
        set(hax(labeli,ndx),prop,axcProps.(prop));
      end
      
      if freezeInfo.isrotated,
        set(hax(labeli,ndx),'CameraUpVectorMode','auto');
      end
      %set(hax(labeli,ndx),'XLim',freezeInfo.xdata,'YLim',freezeInfo.ydata);
      
      h = gobjects(1,nprcs);
      circtheta = linspace(-pi,pi,20);
      xcirc = cos(circtheta);
      ycirc = sin(circtheta);
      for pti = 1:nlandmarks,
        %plot(lpos(pti,1),lpos(pti,2),'+','Color',ptcolors(pti,:));
        for prci = 1:nprcs,
          x = lpos(pti,1)+xcirc*errprctilespts(prci,ndx,pti,labeli,datai);
          y = lpos(pti,2)+ycirc*errprctilespts(prci,ndx,pti,labeli,datai);
          h(prci) = plot(x,y,'-','Color',prccolors(prci,:));
        end
      end
      if labeli == 1,
        title(legendnames{ndx});
      end
      if labeli == 1 && ndx == 1,
        legend(h,num2str(prcs(:)));
      end
    end
    ylabel(hax(labeli,1),labeltypes{labeli});
  end
  
  %savefig(hfigs(datai),sprintf('FlyBubble_GTTrackingError_%sOnFly_%s.fig',savekey,datatypes{datai,1}),'compact');
  if dosavefig,
    if numel(savenames) < datai,
      savenames{datai} = sprintf('%s_GTTrackingError_PrctileOnFly_%s.svg',exptype,datatypes{datai,1});
    end
    saveas(hfigs(datai),fullfile(savedir,savenames{datai}),'svg');
  end
  
end