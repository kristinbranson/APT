function nframes = readLocalTrkFileStatus(obj,filename,partFileIsTextStatus)
  nframes = 0;
  if nargin < 3,
    partFileIsTextStatus = false;
  end
  if ~exist(filename,'file'),
    return;
  end
  if partFileIsTextStatus,
    s = obj.fileContents(filename);
    nframes = TrkFile.getNFramesTrackedPartFile(s);
  end
end
