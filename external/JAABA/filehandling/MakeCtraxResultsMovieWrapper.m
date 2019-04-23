% make results movies
function MakeCtraxResultsMovieWrapper(moviefile,trxfile,xvidfile,movietitle,ctraxresultsmovie_params)


%% ctrax movie parameters
defaulttempdatadir = '/groups/branson/bransonlab/projects/olympiad/TempData_FlyBowlMakeCtraxResultsMovie';
if ~isfield(ctraxresultsmovie_params,'tempdatadir'),
  ctraxresultsmovie_params.tempdatadir = defaulttempdatadir;
elseif isunix
  [status1,res] = unix(sprintf('echo %s',ctraxresultsmovie_params.tempdatadir));
  if status1 == 0,
    ctraxresultsmovie_params.tempdatadir = strtrim(res);
  end
end

if ~exist(ctraxresultsmovie_params.tempdatadir,'dir'),
  [success1,msg1] = mkdir(ctraxresultsmovie_params.tempdatadir);
  if ~success1,
    error('Error making directory %s: %s',ctraxresultsmovie_params.tempdatadir,msg1);
  end
end

%% location of data

[~,avifilestr,ext] = fileparts(xvidfile);
avifile = fullfile(ctraxresultsmovie_params.tempdatadir,[avifilestr,'_temp',ext]);

%% which frames

trx = load_tracks(trxfile);
start_frame = min([trx.firstframe]);
end_frame = max([trx.endframe]);
nframes = end_frame-start_frame + 1;
firstframes_off = min(max(0,round(ctraxresultsmovie_params.firstframes*nframes)),nframes-1);
firstframes_off(ctraxresultsmovie_params.firstframes < 0) = nan;
middleframes_off = round(ctraxresultsmovie_params.middleframes*nframes);
middleframes_off(ctraxresultsmovie_params.middleframes < 0) = nan;
endframes_off = round(ctraxresultsmovie_params.endframes*nframes);
endframes_off(ctraxresultsmovie_params.endframes < 0) = nan;
idx = ~isnan(middleframes_off);
firstframes_off(idx) = ...
  min(nframes-1,max(0,middleframes_off(idx) - ceil(ctraxresultsmovie_params.nframes(idx)/2)));
idx = ~isnan(endframes_off);
firstframes_off(idx) = ...
  min(nframes-1,max(0,endframes_off(idx) - ctraxresultsmovie_params.nframes(idx)));
endframes_off = firstframes_off + ctraxresultsmovie_params.nframes - 1;

firstframes = start_frame + firstframes_off;

%% option to not specify nzoomr, nzoomc

if ischar(ctraxresultsmovie_params.nzoomr) || ischar(ctraxresultsmovie_params.nzoomc),
    
  nflies = zeros(1,nframes);
  for i = 1:numel(trx),
    j0 = trx(i).firstframe-start_frame+1;
    j1 = trx(i).endframe-start_frame+1;
    nflies(j0:j1) = nflies(j0:j1)+1;
  end
  mediannflies = median(nflies);

  if isnumeric(ctraxresultsmovie_params.nzoomr),
    nzoomr = ctraxresultsmovie_params.nzoomr;
    nzoomc = round(mediannflies/nzoomr);
  elseif isnumeric(ctraxresultsmovie_params.nzoomc),
    nzoomc = ctraxresultsmovie_params.nzoomc;
    nzoomr = round(mediannflies/nzoomc);
  else
    nzoomr = ceil(sqrt(mediannflies));
    nzoomc = round(mediannflies/nzoomr);
  end
  ctraxresultsmovie_params.nzoomr = nzoomr;
  ctraxresultsmovie_params.nzoomc = nzoomc;
  
  if iscell(ctraxresultsmovie_params.figpos),  
    [readframe,~,fid] = get_readframe_fcn(moviefile);
    im = readframe(1);
    [nr,nc,~] = size(im);
    
    rowszoom = floor(nr/nzoomr);
    imsize = [nr,nc+rowszoom*nzoomc];
    figpos = str2double(ctraxresultsmovie_params.figpos);
    if isnan(figpos(3)),
      figpos(3) = figpos(4)*imsize(2)/imsize(1);
    elseif isnan(figpos(4)),
      figpos(4) = figpos(3)*imsize(1)/imsize(2);
    end
    ctraxresultsmovie_params.figpos = figpos;
    
    if fid > 1,
      fclose(fid);
    end
  end
  
  
  
end

%% create movie

[succeeded,~,~,height,width]= ...
  make_ctrax_result_movie('moviename',moviefile,'trxname',trxfile,'aviname',avifile,...
  'nzoomr',ctraxresultsmovie_params.nzoomr,'nzoomc',ctraxresultsmovie_params.nzoomc,...
  'boxradius',ctraxresultsmovie_params.boxradius,'taillength',ctraxresultsmovie_params.taillength,...
  'fps',ctraxresultsmovie_params.fps,...
  'maxnframes',ctraxresultsmovie_params.nframes,...
  'firstframes',firstframes,...
  'figpos',ctraxresultsmovie_params.figpos,...
  'movietitle',movietitle,...
  'compression','none',...
  'useVideoWriter',false,...
  'titletext',false,...
  'avifileTempDataFile',[avifile,'-temp'],...
  'dynamicflyselection',true,...
  'doshowsex',true);

if ishandle(1),
  close(1);
end

if ~succeeded,
  error('Failed to create raw avi %s',avifile);
end

%% create subtitle file

subtitlefile = [tempname(ctraxresultsmovie_params.tempdatadir),'.srt'];
fid = fopen(subtitlefile,'w');
dt = [0,ctraxresultsmovie_params.nframes];
ts = cumsum(dt);
for i = 1:numel(dt)-1,
  fprintf(fid,'%d\n',i);
  fprintf(fid,'%s --> %s\n',...
    datestr(ts(i)/ctraxresultsmovie_params.fps/(3600*24),'HH:MM:SS,FFF'),...
    datestr((ts(i+1)-1)/ctraxresultsmovie_params.fps/(3600*24),'HH:MM:SS,FFF'));
  fprintf(fid,'%s, fr %d-%d\n\n',movietitle,...
    firstframes_off(i)+1,...
    endframes_off(i)+1);
end
fclose(fid);

%% compress

tmpfile = [xvidfile,'.tmp'];
newheight = 4*ceil(height/4);
newwidth = 4*ceil(width/4);
% subtitles are upside down, so encode with subtitles and flip, then flip
% again
cmd = sprintf('mencoder %s -o %s -ovc xvid -xvidencopts fixed_quant=4 -vf scale=%d:%d,flip -sub %s -subfont-text-scale 2 -msglevel all=2',...
  avifile,tmpfile,newwidth,newheight,subtitlefile);
status = system(cmd);
if status ~= 0,
  fprintf('*****\n');
  warning('Failed to compress avi to %s',xvidfile);
  fprintf('Need to run:\n');
  fprintf('%s\n',cmd);
  cmd2 = sprintf('mencoder %s -o %s -ovc xvid -xvidencopts fixed_quant=4 -vf flip -msglevel all=2',...
    tmpfile,xvidfile);
  fprintf('then\n');
  fprintf('%s\n',cmd2);
  fprintf('then delete %s %s %s\n',tmpfile,avifile,subtitlefile);
  fprintf('*****\n');
else
  cmd = sprintf('mencoder %s -o %s -ovc xvid -xvidencopts fixed_quant=4 -vf flip -msglevel all=2',...
    tmpfile,xvidfile);
  status = system(cmd);
  if status ~= 0,
    fprintf('*****\n');
    warning('Failed to add subtitles to %s',xvidfile);
    fprintf('Need to run:\n');
    fprintf('%s\n',cmd);
    fprintf('then delete %s %s %s\n',tmpfile,avifile,subtitlefile);
    fprintf('*****\n');    
  else
    delete(tmpfile);
    delete(avifile);
    delete(subtitlefile);
  end
end
