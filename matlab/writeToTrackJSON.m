function writeToTrackJSON(toTrack,jsonfile)

fns = fieldnames(toTrack);

dict = struct;
dict.movfiles = 'movie_files';
dict.trkfiles = 'output_files';
dict.trxfiles = 'trx_files';
dict.cropRois = 'crop_rois';
dict.calibrationfiles = 'calibration_file';
dict.targets = 'targets';
dict.f0s = 'frame0';
dict.f1s = 'frame1';

res = struct;
res.toTrack = cell(numel(toTrack.movfiles),1);

for i = 1:numel(toTrack.movfiles),
  res.toTrack{i} = struct;
  for j = 1:numel(fns),
  
    fn = fns{j};
    x = toTrack.(fn)(i,:);
    fnout = dict.(fn);
    
    if numel(x) == 1,
      res.toTrack{i}.(fnout) = x{1};
    else
      res.toTrack{i}.(fnout) = struct;
      for k = 1:numel(x),
        res.toTrack{i}.(fnout).(sprintf('view%d',k)) = x{k};
      end
    end
  end
  
end

saveJSONfile(res,jsonfile);
