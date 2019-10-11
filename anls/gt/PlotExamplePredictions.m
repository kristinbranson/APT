function [hfigs,savenames] = PlotExamplePredictions(varargin)

[gtdata,gtimdata,nets,legendnames,ptcolors,exptype,...
  conddata,labeltypes,datatypes,...
  figpos,savedir,savenames,dosavefig,...
  reseed,nexamples_random,nexamples_disagree,...
  ms,lw,textcolor,lObj,errnets,doAlignCoordSystem] = myparse(varargin,'gtdata',[],'gtimdata',[],...
  'nets',{},'legendnames',{},'ptcolors',[],'exptype','exp',...
  'conddata',[],'labeltypes',{},'datatypes',{},...
  'figpos',[],...
  'savedir','.',...
  'savenames',{},...
  'dosavefig',false,...
  'reseed',0,...
  'nexamples_random',5,...
  'nexamples_disagree',5,...
  'MarkerSize',8,...
  'LineWidth',2,...
  'textcolor','c',...
  'lObj',[],...
  'errnets',{},...
  'doAlignCoordSystem',false);

assert(~isempty(gtdata));
assert(~isempty(gtimdata));

isshexp = startsWith(exptype,'SH');

vwi = str2double(exptype(end))+1;
if isnan(vwi),
  vwi = 1;
end

if isempty(nets),
  nets = fieldnames(gtdata);
end
if isempty(legendnames),
  legendnames = nets;
end
idxremove = ~isfield(gtdata,nets);
nets(idxremove) = [];
legendnames(idxremove) = [];
if isempty(errnets),
  errnets = nets;
end
nnets = numel(nets);
errnetidx = find(ismember(nets,errnets));

ndatapts = size(gtdata.(nets{1}){end}.labels,1);
if isempty(conddata),
  conddata = struct;
  conddata.data_cond = ones(ndatapts,1);
  conddata.label_cond = ones(ndatapts,1);
  ndatatypes = 1;
  nlabeltypes = 1;
end
nlandmarks = size(gtdata.(nets{1}){end}.labels,2);
d = size(gtdata.(nets{1}){end}.labels,3);

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

if isempty(ptcolors),
  ptcolors = jet(nlandmarks);
end  

if isshexp,
  for ndx = 1:nnets,
    iscpr = ~isempty(strfind(nets{ndx},'cpr'));
    if ~iscpr,
      continue;
    end
    gtdata.(nets{ndx}){end}.pred = cat(1,gtdata.(nets{ndx}){end}.pred(1:3,:,:),nan([1,nlandmarks,2]),gtdata.(nets{ndx}){end}.pred(4:end,:,:));
    gtdata.(nets{ndx}){end}.labels = cat(1,gtdata.(nets{ndx}){end}.labels(1:3,:,:),nan([1,nlandmarks,2]),gtdata.(nets{ndx}){end}.labels(4:end,:,:));
  end
end

if isempty(figpos),
  figpos = [10,10,1332,1468];
end

%% plot example predictions

if ~isempty(reseed),
  rng(reseed);
end
nexamples = nexamples_random + nexamples_disagree;

hfigs = gobjects(ndatatypes,nlabeltypes);

labelscurr = gtdata.(nets{1}){end}.labels;
if doAlignCoordSystem,
  labelscurr = align(labelscurr,gtimdata,lObj,vwi);
end
isbadlabel = false(ndatapts,1);
for i = 1:ndatapts,
  labelcurr = permute(labelscurr(i,:,:),[2,3,1]);
  isbadlabel(i) = any(any(labelcurr < 1 | labelcurr > fliplr(size(gtimdata.ppdata.I{i,vwi}))));
end
if any(isbadlabel),
  warning('%d bad labels ignored',nnz(isbadlabel));
end

for datai = 1:ndatatypes,
  for labeli = 1:nlabeltypes,
    idx = find(ismember(conddata.data_cond,datatypes{datai,2})&ismember(conddata.label_cond,labeltypes{labeli,2})&~isbadlabel);
    if isempty(idx),
      continue;
    end
    hfigs(datai,labeli) = figure;
    set(hfigs(datai,labeli),'Units','pixels','Position',figpos,'Renderer','painters');

    allpreds = nan([size(gtdata.(nets{1}){end}.pred),nnets]);
    
    for neti = 1:nnets,
      
      predscurr = gtdata.(nets{neti}){end}.pred;
      if doAlignCoordSystem || ~isempty(strfind(nets{neti},'cpr')) || ismember(nets{neti},{'Alice','Austin'}),
        predscurr = align(gtdata.(nets{neti}){end}.pred,gtimdata,lObj,vwi);
      end
      allpreds(:,:,:,neti) = predscurr;
    end
    
    if nexamples > numel(idx),
      exampleidx = idx;
      exampleinfo = repmat({'Rand'},[numel(exampleidx),1]);

    else
      medpred = median(allpreds(idx,:,:,errnetidx),4);
      disagreement = max(max(sqrt(sum( (allpreds(idx,:,:,errnetidx)-medpred).^2,3 )),[],4),[],2);
      [sorteddisagreement,order] = sort(disagreement,1,'descend');
      exampleidx_disagree = idx(order(1:nexamples_disagree));
      isselected = false(ndatapts,1);
      isselected(exampleidx_disagree) = true;
      idx = find(ismember(conddata.data_cond,datatypes{datai,2})&ismember(conddata.label_cond,labeltypes{labeli,2})&~isselected&~isbadlabel);
      exampleidx_random = randsample(idx,nexamples_random);
      exampleidx = [exampleidx_random;exampleidx_disagree];
      exampleinfo = repmat({'Rand'},[numel(exampleidx_random),1]);
      for i = 1:nexamples_disagree,
        exampleinfo{end+1} = sprintf('%.1f',sorteddisagreement(i));  %#ok<AGROW>
      end
      
    end
    hax = createsubplots(nexamples,nnets+2,0);%[[.025,0],[.025,0]]);
    hax = reshape(hax,[nexamples,nnets+2]);
    hax = hax';
    for exii = 1:numel(exampleidx),
      exi = exampleidx(exii);
      imagesc(gtimdata.ppdata.I{exi,vwi},'Parent',hax(1,exii));
      axis(hax(1,exii),'image','off');
      hold(hax(1,exii),'on');
      for pti = 1:nlandmarks,
        plot(hax(1,exii),labelscurr(exi,pti,1),labelscurr(exi,pti,2),'+','Color',ptcolors(pti,:),'MarkerSize',ms,'LineWidth',lw);
      end
      text(1,size(gtimdata.ppdata.I{exi,vwi},1)/2,...
        sprintf('%d, %s (Mov %d, Tgt %d, Frm %d)',exi,exampleinfo{exii},...
        -gtimdata.tblPGT.mov(exi),gtimdata.tblPGT.iTgt(exi),gtimdata.tblPGT.frm(exi)),...
        'Rotation',90,'HorizontalAlignment','center','VerticalAlignment','top','Parent',hax(1,exii),'FontSize',6,'Color',textcolor);
    end
    exi = exampleidx(1);
    text(size(gtimdata.ppdata.I{exi,vwi},2)/2,1,'Groundtruth','HorizontalAlignment','center','VerticalAlignment','top','Parent',hax(1,1),'Color',textcolor);
    
    for exii = 1:numel(exampleidx),
      exi = exampleidx(exii);
      imagesc(gtimdata.ppdata.I{exi,vwi},'Parent',hax(2,exii));
      axis(hax(2,exii),'image','off');
      hold(hax(2,exii),'on');
      for pti = 1:nlandmarks,
        plot(hax(2,exii),labelscurr(exi,pti,1),labelscurr(exi,pti,2),'+','Color',ptcolors(pti,:),'MarkerSize',ms,'LineWidth',lw);
      end
      for pti = 1:nlandmarks,
        plot(hax(2,exii),squeeze(allpreds(exi,pti,1,:)),squeeze(allpreds(exi,pti,2,:)),'.','Color',ptcolors(pti,:),'MarkerSize',ms,'LineWidth',lw);
      end
    end
    exi = exampleidx(1);
    text(size(gtimdata.ppdata.I{exi,vwi},2)/2,1,'All','HorizontalAlignment','center','VerticalAlignment','top','Parent',hax(2,1),'Color',textcolor);

    
    for neti = 1:nnets,
      
      predscurr = allpreds(:,:,:,neti);
      
      for exii = 1:numel(exampleidx),
        exi = exampleidx(exii);
        imagesc(gtimdata.ppdata.I{exi,vwi},'Parent',hax(neti+2,exii));
        axis(hax(neti+2,exii),'image','off');
        hold(hax(neti+2,exii),'on');
        for pti = 1:nlandmarks,
          plot(hax(neti+2,exii),predscurr(exi,pti,1),predscurr(exi,pti,2),'+','Color',ptcolors(pti,:),'MarkerSize',ms,'LineWidth',lw);
        end
        err = max(sqrt(sum((predscurr(exi,:,:)-labelscurr(exi,:,:)).^2,3)));
        text(size(gtimdata.ppdata.I{exi,vwi},2)/2,size(gtimdata.ppdata.I{exi,vwi},1),num2str(err),'HorizontalAlignment','center','VerticalAlignment','bottom','Parent',hax(neti+2,exii),'Color',textcolor);
      end
      text(size(gtimdata.ppdata.I{exi,vwi},2)/2,1,legendnames{neti},'HorizontalAlignment','center','VerticalAlignment','top','Parent',hax(neti+2,1),'Color',textcolor);
    end
    colormap(hfigs(datai,labeli),'gray');

    set(hfigs(datai,labeli),'Name',sprintf('Examples: %s, %s, %s',exptype,datatypes{datai,1},labeltypes{labeli,1}));
    drawnow;
    savei = sub2ind([ndatatypes,nlabeltypes],datai,labeli);
    if dosavefig,
      if numel(savenames)<savei || isempty(savenames{savei})
        savenames{savei} = fullfile(savedir,sprintf('%s_Examples_%s_%s.svg',exptype,datatypes{datai,1},labeltypes{labeli,1}));
      end        
      saveas(hfigs(datai,labeli),savenames{savei},'svg');
    end
          
  end
end

function predscurr = align(preds,gtimdata,lObj,vwi)

if isstruct(lObj),
  preProcParams = APTParameters.all2PreProcParams(lObj.trackParams);
else
  preProcParams = lObj.preProcParams;
end

tblP = gtimdata.tblPGT;
isTrx = any(~isnan(tblP.pTrx(:)));
ndatapts = size(preds,1);
predscurr = preds;

% doing this manually, can't find a good function to do it
if isTrx,
  for i = 1:ndatapts,
    pAbs = permute(preds(i,:,:),[2,3,1]);
    x = tblP.pTrx(i,1);
    y = tblP.pTrx(i,2);
    if isnan(x),
      continue;
    end
    T = [1,0,0
      0,1,0
      -x,-y,1];
    theta = tblP.thetaTrx(i);
    if preProcParams.TargetCrop.AlignUsingTrxTheta && ~isnan(theta),
      R = [cos(theta+pi/2),-sin(theta+pi/2),0
        sin(theta+pi/2),cos(theta+pi/2),0
        0,0,1];
    else
      R = eye(3);
    end
    A = T*R;
    tform = maketform('affine',A);
    [pRel(:,1),pRel(:,2)] = ...
      tformfwd(tform,pAbs(:,1),pAbs(:,2));
    pRoi = preProcParams.TargetCrop.Radius+pRel;
    predscurr(i,:,:) = pRoi;
  end
else
  if ismember('roi',gtimdata.ppdata.MD.Properties.VariableNames)
    nvws = size(gtimdata.ppdata.MD.roi,2)/4;
    roi = reshape(gtimdata.ppdata.MD.roi,[ndatapts,4,nvws]);
    roi = roi(:,:,vwi);
    predscurr = preds - reshape(roi(:,[1,3]),[ndatapts,1,2]);
  else
    predscurr = preds;
  end
end
