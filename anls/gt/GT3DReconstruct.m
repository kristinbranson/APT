function gtdata = GT3DReconstruct(gtdatain,gtinfo,varargin)

[isshexp] = myparse_nocheck(varargin,'isshexp',false);

nviews = numel(gtdatain);
% run on all cells of networks
if isstruct(gtdatain{end}) && ~all(isfield(gtdatain{1},{'pred','labels'})),
  gtdata = gtdatain{end};
  netfns = fieldnames(gtdatain{end});
  labels3d = [];
  labels2d = cell(1,nviews);
  for i = 1:numel(netfns),
    netfn = netfns{i};
    for j = 1:numel(gtdatain{end}.(netfn)),
      fprintf('%s %d\n',netfn,j);
      gtdataincurr = cell(1,nviews);
      for vwi = 1:nviews,
        gtdataincurr{vwi} = gtdatain{vwi}.(netfn){j};
      end
      iscpr = contains(netfn,'cpr');
      args = {};
      if ~iscpr && ~isempty(labels3d),
        for vwi = 1:nviews,
          assert(all(labels2d{vwi}(:) == gtdataincurr{vwi}.labels(:)));
        end
        args = {'labels3d',labels3d};
      end
      gtinfocurr = gtinfo;
      % missing label for stephen's data & cpr
      if isshexp && iscpr,
        gtinfocurr.movieidx(4) = [];
      end
      
      gtdatacurr = GT3DReconstruct(gtdataincurr,gtinfocurr,args{:},'iscropped',~iscpr);
      gtdata.(netfn){j} = gtdatacurr;
      if ~iscpr && isempty(labels3d),
        labels3d = gtdatacurr.labels;
        labels2d = cell(1,nviews);
        for vwi = 1:nviews,
          labels2d{vwi} = gtdataincurr{vwi}.labels;
        end
      end
    end
  end
  return;
end

[labels3dref,iscropped] = myparse(varargin,'labels3d',[],'iscropped',true);

dolabels = isempty(labels3dref);

[ndatapts,nlandmarks,din] = size(gtdatain{end}.labels);
gtdata = gtdatain{end};
if dolabels,
  labelsout = nan([ndatapts,nlandmarks,3]);
else
  gtdata.labels = labels3dref;
end
predout = nan([ndatapts,nlandmarks,3]);

parfor i = 1:size(gtdatain{end}.pred,1),
  movieidx = gtinfo.movieidx(i);
  caldata = gtinfo.viewCalibrationDataGT{movieidx};
  reconstructfun = get_reconstruct_fcn(caldata);
  % uv is d=2 x nviews=2 x n
  if dolabels,
    labels = nan([2,nviews,nlandmarks]);
  end
  pred = nan([2,nviews,nlandmarks]);
  for vwi = 1:nviews,
    if iscropped && ~isempty(gtinfo.cropInfo{movieidx}),
      offx = gtinfo.cropInfo{movieidx}(vwi).roi(1);
      offy = gtinfo.cropInfo{movieidx}(vwi).roi(3);
    else
      offx = 0;
      offy = 0;
    end
    if dolabels,
      labels(:,vwi,:) = permute(gtdatain{vwi}.labels(i,:,:),[3,1,2]);
      labels(1,vwi,:) = labels(1,vwi,:) + offx;
      labels(2,vwi,:) = labels(2,vwi,:) + offy;
    end
    
    pred(:,vwi,:) = permute(gtdatain{vwi}.pred(i,:,:),[3,1,2]);
    pred(1,vwi,:) = pred(1,vwi,:) + offx;
    pred(2,vwi,:) = pred(2,vwi,:) + offy;
  end
  if dolabels
    labels3d = reconstructfun(labels);
    labelsout(i,:,:) = labels3d';
  end
  pred3d = reconstructfun(pred);
  predout(i,:,:) = pred3d';
end
if dolabels,
  gtdata.labels = labelsout;
end
gtdata.pred = predout;