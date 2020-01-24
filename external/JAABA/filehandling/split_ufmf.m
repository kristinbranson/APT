function outfiles = split_ufmf(infile,outfilestr,chunksize,varargin)

[startframe,endframe,dooverwrite] = ...
  myparse(varargin,'startframe',1,'endframe',inf,'dooverwrite',false);

% read in the header
header = ufmf_read_header(infile);
endframe = min(endframe,header.nframes);
noutfiles = ceil((endframe-startframe+1)/chunksize);

% find the end of the header
headerstart = 0;
% loc of something after the header
headerend = min(min(header.mean2file),min(header.frame2file));

% read in the header
fseek(header.fid,headerstart,'bof');
headerbytes = fread(header.fid,headerend-headerstart);

outfiles = cell(1,noutfiles);
for outi = 1:noutfiles,
  
  t0 = startframe + (outi-1)*chunksize;
  t1 = min(endframe,t0+chunksize-1);
  
  outfiles{outi} = sprintf('%s_%02d.ufmf',outfilestr,outi);
  if exist(outfiles{outi},'file') && ~dooverwrite,
    res = input(sprintf('File %s exists. Overwrite? (y/N): ',outfiles{outi}),'s');
    if ~strcmpi(res,'y'),
      fclose(header.fid);
      return;
    end
  end
  
  fidout = fopen( outfiles{outi}, 'wb' , 'ieee-le');
  assert(fidout > 0);
  
  % write the header
  fwrite(fidout,headerbytes);
  
  lastmean = nan;
  
  index = struct;
  index.frame = struct;
  index.frame.loc = [];
  index.frame.timestamp = header.timestamps(t0:t1);
  index.keyframe = struct;
  index.keyframe.mean = struct;
  index.keyframe.mean.loc = [];
  index.keyframe.mean.timestamp = [];
  

  % write the frames
  for t = t0:t1,

    % write a mean
    if header.frame2mean(t) ~= lastmean,
      meanstart = header.mean2file(header.frame2mean(t));
      meanend = min([min(header.mean2file(header.mean2file>meanstart)),...
        min(header.frame2file(header.frame2file>meanstart)),...
        header.indexloc]);
      assert(~isempty(meanend) && ~isnan(meanend));
      index.keyframe.mean.loc(end+1) = ftell(fidout);
      index.keyframe.mean.timestamp(end+1) = header.mean_timestamps(header.frame2mean(t));
      fseek(header.fid,meanstart,'bof');
      fwrite(fidout,fread(header.fid,meanend-meanstart));
      lastmean = header.frame2mean(t);
    end
    
    framestart = header.frame2file(t);
    frameend = min([min(header.mean2file(header.mean2file>framestart)),...
        min(header.frame2file(header.frame2file>framestart)),...
        header.indexloc]);
    assert(~isempty(frameend) && ~isnan(frameend));
    fseek(header.fid,framestart,'bof');
    index.frame.loc(end+1) = ftell(fidout);
    fwrite(fidout,fread(header.fid,frameend-framestart));
  end
  
  % write the index
  wrapupUFMF(fidout,index,header.indexlocloc);
  fclose(fidout);
  
end
