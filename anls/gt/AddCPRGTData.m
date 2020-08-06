function gtdata = AddCPRGTData(gtdata,gtfile_cpr,labeledpos,vwi)

if ~exist('vwi','var'),
  vwi = 1;
end

nets = fieldnames(gtdata);
fns0 = fieldnames(gtdata);
nmodels0 = numel(gtdata.(fns0{1}));
gtdata_cpr = load(gtfile_cpr);
if isfield(gtdata_cpr,'xvRes'),
    
  % sanity check - labels match up
  netlabels = gtdata.(nets{1}){end}.labels;
  nlandmarks = size(netlabels,2);
  cprlabels = nan(size(netlabels));
  nvws = size(gtdata_cpr.xvRes.p,2)/2/nlandmarks;
  for i = 1:numel(labeledpos),
    sz = size(labeledpos{i});
    assert(sz(1)==nlandmarks*nvws);
    labeledpos{i} = reshape(labeledpos{i},[nlandmarks,nvws,sz(2:end)]);
  end
  for i = 1:size(gtdata_cpr.xvRes,1),
    cprlabels(i,:,:) = labeledpos{gtdata_cpr.xvRes.mov(i)}(:,vwi,:,gtdata_cpr.xvRes.frm(i),gtdata_cpr.xvRes.iTgt(i));
  end
  cprlabels0 = cprlabels-cprlabels(:,1,:);
  netlabels0 = netlabels-netlabels(:,1,:);
  
  if max(max(max(abs(netlabels0-cprlabels0),[],2),[],3))>.01,
    order = nan(1,size(cprlabels0,1));
    for i = 1:size(cprlabels0,1),
      d = sum(sum(abs(cprlabels0(i,:,:)-netlabels0),2),3);
      d(order(1:i-1)) = nan;
      if d(i) < .001,
        order(i) = i;
      elseif nnz(d<.001) == 1,
        [mind,j] = min(d);
        order(i) = j;
      else
        [mind,j] = min(d);
        if isnan(mind),
          order(i) = i;
          warning('mind is nan for i = %d\n',i);
        else
          order(i) = j;
          %warning('min distance between cpr and net labels is %.1f for i = %d\n',mind,i);
        end
      end
    end
    assert(numel(unique(order))==numel(order));
    for i = 1:numel(nets),
      for j = 1:numel(gtdata.(nets{i})),
        assert(all(netlabels(:) == gtdata.(nets{i}){j}.labels(:)));
        gtdata.(nets{i}){j}.labels = gtdata.(nets{i}){j}.labels(order,:,:);
        gtdata.(nets{i}){j}.pred = gtdata.(nets{i}){j}.pred(order,:,:);
      end
    end
    netlabels = gtdata.(nets{1}){end}.labels;
    netlabels0 = netlabels-netlabels(:,1,:);
  end
  
  %assert(max(max(max(abs(netlabels0-cprlabels0),[],2),[],3))<.01);
  gtdata.cpropt{1} = struct;
  gtdata.cpropt{1}.labels = cprlabels;
  pTrk = reshape(gtdata_cpr.xvRes.pTrk,[size(gtdata_cpr.xvRes.pTrk,1),nlandmarks,nvws,2]);
  gtdata.cpropt{1}.pred = reshape(pTrk(:,:,vwi,:),size(cprlabels));
    
else
  %newfns = setdiff(fieldnames(gtdata_cpr),fieldnames(gtdata));
  newfns = intersect(fieldnames(gtdata_cpr),{'cpropt','cprqck'});
  for i = 1:numel(newfns),
    gtdata.(newfns{i}) = gtdata_cpr.(newfns{i});
    if nmodels0 == 1,
      gtdata.(newfns{i}) = gtdata.(newfns{i})(end);
    end
  end
end
