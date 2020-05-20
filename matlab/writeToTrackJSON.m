function writeToTrackJSON(toTrack,jsonfile)


dict = struct;
dict.movfiles = 'movie_files';
dict.trkfiles = 'output_files';
dict.trxfiles = 'trx_files';
dict.cropRois = 'crop_rois';
dict.calibrationfiles = 'calibration_file';
dict.targets = 'targets';
dict.f0s = 'frame0';
dict.f1s = 'frame1';
nview = size(toTrack.movfiles,2);

fns = fieldnames(dict);

res = struct;
res.toTrack = cell(size(toTrack.movfiles,1),1);

for i = 1:size(toTrack.movfiles,1),
  res.toTrack{i} = struct;
  for j = 1:numel(fns),
  
    fn = fns{j};
    x = toTrack.(fn)(i,:);
    fnout = dict.(fn);
    
    doignore = false(1,numel(x));
    for k = 1:numel(x),
      if iscell(x) && isempty(x{k}),
        doignore(k) = true;
      else
        if iscell(x),
          x1 = x{k};
        else
          x1 = x(k);
        end
        if isnumeric(x1) && all(isnan(x1)),
          doignore(k) = true;
        elseif strcmpi(fn,'f0s') && x1 == 1,
          doignore(k) = true;
        elseif strcmpi(fn,'f1s') && isinf(x1),
          doignore(k) = true;
        end

      end
    end
    
    if isempty(x),
    elseif numel(x) == 1
      if ~doignore,
        if iscell(x),
          res.toTrack{i}.(fnout) = x{1};
        else
          res.toTrack{i}.(fnout) = x;
        end
      end
    elseif numel(x) == nview,
      res.toTrack{i}.(fnout) = struct;
      for k = 1:nview,
        if doignore, continue; end
        if iscell(x),
          res.toTrack{i}.(fnout).(sprintf('view%d',k)) = x{k};
        else
          res.toTrack{i}.(fnout).(sprintf('view%d',k)) = x(k);
        end
      end
    else
      error('Size of %s does not match number of views',fn);
    end
  end
  
end

if size(toTrack.movfiles,1) == 1,
  res.toTrack = res.toTrack{1};
end
saveJSONfile(res,jsonfile);
