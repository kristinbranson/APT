function [err_train,err_test,err_train_stats,err_test_stats,delta_train,delta_test] = ComputeTrainTestError(allpostdata,firstframes,labeldata,varargin)

prctiles_compute = myparse(varargin,'prctiles_compute',[25,50:100]);

isdone = false;
for i = 1:numel(allpostdata),
  for j = 1:numel(allpostdata{i}),
    if ~isempty(allpostdata{i}{j}),
      algorithms = fieldnames(allpostdata{i}{j});
      d = size(allpostdata{i}{j}.(algorithms{1}).x,3);
      isdone = true;
      break;
    end
  end
  if isdone,
    break;
  end
end

predpose_test = struct;
for i = 1:numel(algorithms),
  predpose_test.(algorithms{i}) = nan(size(labeldata.truepose_test));
end

nmovies = numel(allpostdata);
for mi = 1:nmovies,
  if isempty(allpostdata{mi}),
    continue;
  end
  for fly = 1:numel(allpostdata{mi}),
    if isempty(allpostdata{mi}{fly}),
      continue;
    end
    %firstframe = td.trx(fly).firstframe;
    firstframe = firstframes{mi}(fly);
    idxcurr = find(labeldata.movieidxtest==mi & labeldata.flytest == fly);
    if isempty(idxcurr),
      continue;
    end

    ts = labeldata.ttest(idxcurr);

    for j = 1:numel(algorithms),
      algorithm = algorithms{j};
      
      if ~isfield(allpostdata{mi}{fly},algorithm),
        continue;
      end
      predpose_test.(algorithm)(idxcurr,:,:) = allpostdata{mi}{fly}.(algorithm).x(ts-firstframe+1,:,:);
      
    end
    
  end
end

predpose_train = struct;
for i = 1:numel(algorithms),
  predpose_train.(algorithms{i}) = nan(size(labeldata.truepose_train));
end

for mi = 1:nmovies,
  if isempty(allpostdata{mi}),
    continue;
  end
  for fly = 1:numel(allpostdata{mi}),
    if isempty(allpostdata{mi}{fly}),
      continue;
    end
    %firstframe = td.trx(fly).firstframe;
    firstframe = firstframes{mi}(fly);
    idxcurr = find(labeldata.movieidxtrain==mi & labeldata.flytrain == fly);
    if isempty(idxcurr),
      continue;
    end

    ts = labeldata.ttrain(idxcurr);

    for j = 1:numel(algorithms),
      algorithm = algorithms{j};
      
      if ~isfield(allpostdata{mi}{fly},algorithm),
        continue;
      end
      predpose_train.(algorithm)(idxcurr,:,:) = allpostdata{mi}{fly}.(algorithm).x(ts-firstframe+1,:,:);
      
    end
    
  end
end

delta_train = struct;
delta_test = struct;
for i = 1:numel(algorithms),
  algorithm = algorithms{i};
  delta_test.(algorithm) = nan(size(labeldata.truepose_test));
  delta_train.(algorithm) = nan(size(labeldata.truepose_train));
  tmpidx = find(all(all(~isnan(predpose_train.(algorithm)),2),3));
  delta_train.(algorithm)(tmpidx,:,:) = labeldata.truepose_train(tmpidx,:,:)-predpose_train.(algorithm)(tmpidx,:,:);
  tmpidx = find(all(all(~isnan(predpose_test.(algorithm)),2),3));
  delta_test.(algorithm)(tmpidx,:,:) = labeldata.truepose_test(tmpidx,:,:)-predpose_test.(algorithm)(tmpidx,:,:);
end

err_train = struct;
err_test = struct;
for i = 1:numel(algorithms),
  algorithm = algorithms{i};
  err_train.(algorithm) = sqrt(sum(delta_train.(algorithm).^2,3));
  err_test.(algorithm) = sqrt(sum(delta_test.(algorithm).^2,3));
end

err_train_stats = struct;
err_test_stats = struct;
for i = 1:numel(algorithms),
  algorithm = algorithms{i};
  err_train_stats.(algorithm) = struct;
  err_train_stats.(algorithm).prctiles_compute = prctiles_compute;
  err_train_stats.(algorithm).prctiles_perpart = prctile(err_train.(algorithm),prctiles_compute,1);
  err_train_stats.(algorithm).prctiles_worstpart = prctile(max(err_train.(algorithm),[],2),prctiles_compute,1);
  err_train_stats.(algorithm).prctiles_avepart = prctile(mean(err_train.(algorithm),2),prctiles_compute,1);

  err_test_stats.(algorithm) = struct;
  err_test_stats.(algorithm).prctiles_compute = prctiles_compute;
  err_test_stats.(algorithm).prctiles_perpart = prctile(err_test.(algorithm),prctiles_compute,1);
  err_test_stats.(algorithm).prctiles_worstpart = prctile(max(err_test.(algorithm),[],2),prctiles_compute,1);
  err_test_stats.(algorithm).prctiles_avepart = prctile(nanmean(err_test.(algorithm),2),prctiles_compute,1);

end
