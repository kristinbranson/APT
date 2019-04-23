function [outfiles,outindexfiles] = split_mjpg(infile,inindexfile,outfilestr,chunksize,varargin)

[startframe,endframe,dooverwrite] = ...
  myparse(varargin,'startframe',1,'endframe',inf,'dooverwrite',false);

% read in the header
fprintf('Reading index file %s...\n',inindexfile);
header = ReadIndexedMJPGHeader(infile,inindexfile);
fid = fopen(infile,'r');

endframe = min(endframe,header.nframes);
noutfiles = ceil((endframe-startframe+1)/chunksize);

outfiles = cell(1,noutfiles);
outindexfiles = cell(1,noutfiles);
for outi = 1:noutfiles,
  
  t0 = startframe + (outi-1)*chunksize;
  t1 = min(endframe,t0+chunksize-1);
  
  outfiles{outi} = sprintf('%s_%02d.mjpg',outfilestr,outi);
  outindexfiles{outi} = sprintf('%s_%02d.txt',outfilestr,outi);

  fprintf('%d / %d: Outputting frames %d to %d to %s and %s\n',outi,noutfiles,...
    t0,t1,outfiles{outi},outindexfiles{outi});
  tic;
  
  if exist(outfiles{outi},'file') && ~dooverwrite,
    res = input(sprintf('File %s exists. Overwrite? (y/N): ',outfiles{outi}),'s');
    if ~strcmpi(res,'y'),
      fclose(fid);
      return;
    end
  end
  
  fidout = fopen( outfiles{outi}, 'wb' , 'ieee-le');
  assert(fidout > 0);

  % write the frames
  for t = t0:t1,

    if toc > 10,
      tic;
      fprintf('  %.1f%% complete\n',(t-t0)/(t1-t0+1)*100);
    end
    if t == 1,
      framestart = 0;
    else
      framestart = header.frameend2file(t-1);
    end
    frameend = header.frameend2file(t);
    if t == t0,
      fseek(fid,framestart,'bof');
    end
    fwrite(fidout,fread(fid,frameend-framestart));
  end
  fclose(fidout);

  % write the index
  indexfidout = fopen(outindexfiles{outi},'w');
  if t0 == 1,
    startloc = 0;
  else
    startloc = header.frameend2file(t0-1);
  end
  indexdata = [0:t1-t0;...
    header.timestamp(t0:t1);...
    header.frame2file(t0:t1)-startloc;...
    header.frameend2file(t0:t1)-startloc];
  fprintf(indexfidout,'%f %f %f %f\n',indexdata);
  fclose(indexfidout);
  fprintf('  100.0%% complete\n');
  
end
