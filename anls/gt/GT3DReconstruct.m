function gtdata = GT3DReconstruct(gtdatain,gtinfo,varargin)

[isshexp,istime] = myparse_nocheck(varargin,'isshexp',false,'istime',false);

nviews = numel(gtdatain);
% run on all cells of networks
if isstruct(gtdatain{end}) && ~all(isfield(gtdatain{1},{'pred','labels'})),
  gtdata = gtdatain{end};
  netfns = fieldnames(gtdatain{end});
  labels3d = [];
  labels2d = cell(1,nviews);
  for i = 1:numel(netfns),
    netfn = netfns{i};
    if istime,
      % find all unique timestamps
      dts = cell(1,nviews);
      for vwi = 1:nviews,
        for j = 1:numel(gtdatain{vwi}.(netfn)),
          ts_seconds = cellfun(@(x) x.model_timestamp,gtdatain{vwi}.(netfn));
          dts{vwi} = ts_seconds-ts_seconds(1);
        end
      end
      % find last timestamp for each view before each timestamp
      alldts = unique(cat(2,dts{:}));
      nmodels = numel(alldts);
      modelidx = nan(nviews,nmodels);
      modeldts = nan(nviews,nmodels);
      for j = 1:numel(alldts),
        for vwi = 1:nviews,
          idxcurr = find(dts{vwi}<=alldts(j));
          [modeldts(vwi,j),k] = max(dts{vwi}(idxcurr));
          modelidx(vwi,j) = idxcurr(k);
        end
      end
    else
      nmodels = numel(gtdatain{1}.(netfn));
      for vwi = 2:nviews,
        assert(nmodels == numel(gtdatain{vwi}.(netfn)));
      end
    end
    for j = 1:nmodels,
      fprintf('%s %d\n',netfn,j);
      gtdataincurr = cell(1,nviews);
      for vwi = 1:nviews,
        if istime,
          gtdataincurr{vwi} = gtdatain{vwi}.(netfn){modelidx(vwi,j)};
          totaltime = sum(modeldts(:,j));
        else
          gtdataincurr{vwi} = gtdatain{vwi}.(netfn){j};
        end
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
      if istime,
        gtdatacurr.model_timestamp = totaltime;
      end
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

for vwi = 1:nviews,
  nbad = nnz(any(any(isnan(gtdatain{vwi}.pred),2),3));
  if nbad > 0,
    fprintf('Warning: view %d, %d / %d predictions contain nans\n',vwi,nbad,size(gtdatain{vwi}.pred,1));
  end
end

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
  okidx = all(all(~isnan(pred),1),2);
  pred3d = nan(3,nlandmarks);
  if any(okidx),
    pred3d(:,okidx) = reconstructfun(pred(:,:,okidx));
  end
  predout(i,:,:) = pred3d';
end
if dolabels,
  gtdata.labels = labelsout;
end
gtdata.pred = predout;