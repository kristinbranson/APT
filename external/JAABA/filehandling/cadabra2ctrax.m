% [succeeded,outmatname,trx] = cadabra2ctrax(featname,roiname,moviename,[outmatname],[doflipup],[dofliplr],[rot])
% inputs cadabra mat file output names, loads in the trajectories, and
% converts to ctrax trx format. these trx will be saved to the file
% outmatname (user will be prompted for this name if none given). 
% inputs:
% featname: [optional] "feat" output by cadabra. if not input, user will be
% prompted for file. 
% roiname: [optional] "roi" output by cadabra. if not input, user will be
% prompted for file.
% moviename: name of movie tracked
% outmatname: [optional] mat file to save trx trajectories to. user will be
% prompted for file name if none given
% doflipud: [optional] whether to flip the trajectories vertically across
% the middle of the video. Default: false.
% dofliplr: [optional] whether to flip the trajectories horizontally across
% the center of the video. Default: false.
% rot: [optional] degrees to rotate the trajectories by around the center
% of the video. Default: 0. 

function [succeeded,outmatname,trx] = cadabra2ctrax(featname,roiname,moviename,outmatname,doflipud,dofliplr,rot,movieinfo)

% initialize outputs
succeeded = false;
if ~exist('outmatname','var'),
  outmatname = '';
end
trx = [];

if ~exist('featname','var') || isempty(featname),
  helpmsg = 'Choose CADABRA feat mat file';
  [featname,featpath] = uigetfilehelp('*.mat','Choose feat file','','helpmsg',helpmsg);
  if ~ischar(featname),
    return;
  end
  featname = fullfile(featpath,featname);
end

if ~exist('roiname','var') || isempty(roiname),
  helpmsg = sprintf('Choose CADABRA roi mat file corresponding to feat file %s',featname);
  roiname = strrep(featname,'feat','roi');
  [roiname,roipath] = uigetfilehelp('*.mat','Choose feat file',roiname,'helpmsg',helpmsg);
  if ~ischar(roiname),
    return;
  end
  roiname = fullfile(roipath,roiname);
end

if ~exist('moviename','var') || isempty(moviename),
  helpmsg = sprintf('Choose movie file corresponding to feat file %s',featname);
  moviename = strrep(featname,'_feat.mat','.avi');
  [moviename,moviepath] = uigetfilehelp('*.avi','Choose movie file',moviename,'helpmsg',helpmsg);
  if ~ischar(moviename),
    return;
  end
  moviename = fullfile(moviepath,moviename);
end

if ~exist('doflipud','var'),
  doflipud = false;
end

if ~exist('dofliplr','var'),
  dofliplr = false;
end

if ~exist('rot','var'),
  rot = 0;
end

if rot ~= 0,
  phi = rot*pi/180;
  R = [cos(rot),sin(rot);-sin(rot),cos(rot)];
end


% load in data
feat = load(featname);
roi = load(roiname);

if ~exist('movieinfo','var') || isempty(movieinfo),

  % read fps, frame size, moviename is either the name of the movie or an mmreader object
  if ~ischar(moviename),
    readerobj = moviename;
    moviename = fullfile(get(readerobj,'Path'),get(readerobj,'Name'));
  else
    readerobj = mmreader(moviename);
  end
  fps = get(readerobj,'FrameRate')+0;
  movieheight = get(readerobj,'Height')+0;
  moviewidth = get(readerobj,'Width')+0;
else
  fps = movieinfo.fps;
  movieheight = movieinfo.height;
  moviewidth = movieinfo.width;
end
movieheight_mm = movieheight * roi.scale.y;
moviewidth_mm = moviewidth * roi.scale.x;

% scale, pxpermm
pxpermm = 1/mean([roi.scale.x,roi.scale.y]);

% allocate
c = cell(1,2);
trx = struct('x',c,'y',c,'theta',c,'a',c,'b',c,...
  'id',c,'moviename',c,'firstframe',c,'arena',c,...
  'nframes',c,'endframe',c,'pxpermm',c,'fps',c,'x_mm',c,...
  'y_mm',c,'a_mm',c,'b_mm',c,'dt',c);
obj = [feat.fly_feat.obj1,feat.fly_feat.obj2];
firstframeoff = feat.fly_feat.frame(1) - 1;
arena = struct('x',nan,'y',nan,'r',nan);
for fly = 1:2,
  
  % frames for which fly is tracked
  
  % (x,y) = (0,0) for untracked frames
  badframes = obj(fly).pos_x == 0 & obj(fly).pos_y == 0;
  lastframe = find(~badframes,1,'last');
  firstframe = find(~badframes,1);
  if isempty(lastframe), 
    firstframe = 1;
    lastframe = 0;
  end
  nframes = lastframe - firstframe + 1;
  
  % allocate
  trx(fly).x = nan(1,nframes);
  trx(fly).y = nan(1,nframes);
  trx(fly).theta = nan(1,nframes);
  trx(fly).a = nan(1,nframes);
  trx(fly).b = nan(1,nframes);
  trx(fly).xwingl = nan(1,nframes);
  trx(fly).ywingl = nan(1,nframes);
  trx(fly).xwingr = nan(1,nframes);
  trx(fly).ywingr = nan(1,nframes);
  trx(fly).x_mm = nan(1,nframes);
  trx(fly).y_mm = nan(1,nframes);
  trx(fly).a_mm = nan(1,nframes);
  trx(fly).b_mm = nan(1,nframes);
  trx(fly).dt = nan(1,nframes-1);
  
  % store parameters
  trx(fly).moviename = moviename;
  trx(fly).firstframe = firstframe + firstframeoff;
  trx(fly).endframe = lastframe + firstframeoff;
  trx(fly).nframes = nframes;
  trx(fly).pxpermm = pxpermm;
  trx(fly).fps = fps;
  trx(fly).id = fly;
  trx(fly).arena = arena;
  %trx(fly).f2i = @(f) f - trx(fly).firstframe + 1;
  
  % store data
  idx = feat.fly_feat.frame(firstframe:lastframe)-trx(fly).firstframe + 1;
  % we use the quarter major, minor axis length
  trx(fly).a_mm(idx) = obj(fly).FLength/4;
  % maybe area is major/2 * minor/2 * pi, store quarter minor axis length
  trx(fly).b_mm(idx) = (obj(fly).FArea./(obj(fly).FLength/2)/pi)/2;
  % convert degrees to radians
  trx(fly).theta(idx) = obj(fly).headdir*pi/180;
  
  % convert mm to px, incorporate offset
  trx(fly).x(idx) = obj(fly).pos_x*pxpermm + roi.ROI.cols(1) - 1;
  trx(fly).y(idx) = obj(fly).pos_y*pxpermm + roi.ROI.rows(1) - 1 - 1;
  trx(fly).x_mm = trx(fly).x / pxpermm;
  trx(fly).y_mm = trx(fly).y / pxpermm;
  trx(fly).a = trx(fly).a_mm*pxpermm;
  trx(fly).b = trx(fly).b_mm*pxpermm;
    
  phil = obj(fly).phil*pi/180;
  wingll = obj(fly).wingll*pxpermm;
  xwl = trx(fly).x(idx) + wingll.*cos(-phil-trx(fly).theta(idx)-pi);
  ywl = trx(fly).y(idx) + wingll.*sin(-phil-trx(fly).theta(idx));
  phir = obj(fly).phir*pi/180;
  winglr = obj(fly).winglr*pxpermm;
  xwr = trx(fly).x(idx) + winglr.*cos(phir-trx(fly).theta(idx)-pi);
  ywr = trx(fly).y(idx) + winglr.*sin(phir-trx(fly).theta(idx));
  trx(fly).xwingl(idx) = xwl;
  trx(fly).ywingl(idx) = ywl;
  trx(fly).xwingr(idx) = xwr;
  trx(fly).ywingr(idx) = ywr;
  trx(fly).xwingl_mm = trx(fly).xwingl / pxpermm;
  trx(fly).ywingl_mm = trx(fly).ywingl / pxpermm;
  trx(fly).xwingr_mm = trx(fly).xwingr / pxpermm;
  trx(fly).ywingr_mm = trx(fly).ywingr / pxpermm;
  
  
  trx(fly).dt(idx(1:end-1)) = diff(feat.fly_feat.time(firstframe:lastframe));
  
  % flipud if necessary
  if doflipud,
    trx(fly).y_mm = movieheight_mm - trx(fly).y_mm;
    trx(fly).y = movieheight - trx(fly).y;
    trx(fly).theta = -trx(fly).theta;
  end

  % fliplr if necessary
  if dofliplr,
    trx(fly).x_mm = moviewidth_mm - trx(fly).x_mm;
    trx(fly).x = moviewidth - trx(fly).x;
    trx(fly).theta = modrange(pi-trx(fly).theta,-pi,pi);
  end
  
  % rotate if necessary
  if rot ~= 0,
    trx(fly).x_mm = (trx(fly).x_mm - moviewidth_mm/2)*R + moviewidth_mm/2;
    trx(fly).y_mm = (trx(fly).y_mm - movieheight_mm/2)*R + movieheight_mm/2;
    trx(fly).x = (trx(fly).x - moviewidth/2)*R + moviewidth/2;
    trx(fly).y = (trx(fly).y - movieheight/2)*R + movieheight/2;
    trx(fly).theta = modrange(trx(fly).theta + phi,-pi,pi);
  end
  
end

if ~exist('outmatname','var') || isempty(outmatname),
  [pathstr,name] = fileparts(featname);
  outname = strrep(name,'_feat','');
  outmatname = fullfile(pathstr,[outname,'_trx.mat']);
  helpmsg = {};
  helpmsg{1} = 'Choose mat file to save Ctrax version of trajectories loaded from:';
  helpmsg{2} = sprintf('CADABRA feat file: %s',featname);
  helpmsg{3} = sprintf('CADABRA roi file: %s',roiname);
  [outmatname,outmatpath] = uiputfilehelp('*.mat',sprintf('Save ctrax version of %s',name),outmatname,'helpmsg',helpmsg);
  if ~ischar(outmatname),
    outmatname = '';
    return;
  end
  outmatname = fullfile(outmatpath,outmatname);
  fprintf('Saving to %s...\n',outmatname);
end

save(outmatname,'trx');
succeeded = true;