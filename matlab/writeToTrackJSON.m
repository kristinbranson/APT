function writeToTrackJSON(toTrack,jsonfile)


dict = struct;
dict.movfiles = 'movie_files';
dict.trkfiles = 'output_files';
%dict.detectfiles = 'detect_files';
dict.trxfiles = 'trx_files';
dict.cropRois = 'crop_rois';
dict.calibrationfiles = 'calibration_file';
dict.targets = 'targets';
dict.f0s = 'frame0';
dict.f1s = 'frame1';
nview = size(toTrack.movfiles,2);

fns = fieldnames(dict);

res = struct;
nmovset = size(toTrack.movfiles,1);
res.toTrack = cell(nmovset,1);

for imovset = 1:nmovset
  res.toTrack{imovset} = struct;
  for j = 1:numel(fns),
  
    fn = fns{j};
    x = toTrack.(fn)(imovset,:);
    fnout = dict.(fn);
    
    doignoreperval = false(1,numel(x));
    for k = 1:numel(x),
      if iscell(x) && isempty(x{k}),
        doignoreperval(k) = true;
      else
        if iscell(x),
          x1 = x{k};
        else
          x1 = x(k);
        end
        if isnumeric(x1) && all(isnan(x1)),
          doignoreperval(k) = true;
        elseif strcmpi(fn,'f0s') && x1 == 1,
          doignoreperval(k) = true;
        elseif strcmpi(fn,'f1s') && isinf(x1),
          doignoreperval(k) = true;
        end
      end
    end
    
    if isempty(x),
      % none
    elseif numel(x) == 1
      if ~doignoreperval,
        if iscell(x),
          res.toTrack{imovset}.(fnout) = x{1};
        else
          res.toTrack{imovset}.(fnout) = x;
        end
      end
    elseif numel(x) == nview,
      if any(~doignoreperval)
        res.toTrack{imovset}.(fnout) = struct;
        for k = 1:nview,
          if doignoreperval(k), continue; end
          if iscell(x),
            res.toTrack{imovset}.(fnout).(sprintf('view%d',k)) = x{k};
          else
            res.toTrack{imovset}.(fnout).(sprintf('view%d',k)) = x(k);
          end
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
