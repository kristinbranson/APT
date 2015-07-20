% data = createdata_perfile(moviename,matname,annname)
%
% inputs:
% moviename: name of movie file
% matname: name of matfile in which tracks are stored
% annname: name of annotation file in which parameters are written
%
% outputs:
% data: array of structures of size 1 x nflies
% data(i) has the following fields:
%   x: 1 x nframes(i) array containing the x-position of the fly over its lifetime
%   y: 1 x nframes(i) array containing the y-position of the fly over its lifetime
%   a: 1 x nframes(i) array containing the major axis length/4 of the fly over its lifetime
%   b: 1 x nframes(i) array containing the minor axis length/4 of the fly over its lifetime
%   theta: 1 x nframes(i) array containing the orientation of the fly over its lifetime
%   id: scalar containing the identity of this fly given by mtrax
%   moviename: moviename input
%   firstframe: frame of this movie on which the track begins. if
%   track begins at start of movie, firstframe will be 1
%
% This function loads matname and restructures the data into a more
% usable form.
% The annotation file is only used to (attempt to) read in the
% arena position to normalize the data in this movie so that it is
% compatible with data in other movies. If the arena is not defined
% in the annotation file, then it is detected in the first frame of
% moviename. Normalization consists of translating to make the
% center of the arena the origin and scaling so that the radius of
% the arena is 500. 
%
% dependencies:
% read_ann.m used to read annotation file
% detectarena.m and its dependencies used to detect the arena
% fmf_read.m used to read the first movie frame
%
function [data,x_pos,y_pos,maj_ax,min_ax,newidentity] = createdata_perfile(moviename,matname,annname,donormalize)

if ~exist('donormalize','var'),
  donormalize = true;
end

% read arena parameters
if donormalize,
  if exist(annname,'file'),
    [arena_center_x,arena_center_y,arena_radius] = ...
      read_ann(annname,'arena_center_x','arena_center_y','arena_radius');
  else
    arena_center_x = [];
  end;
  if isempty(arena_center_x),
    
    % read a frame of the movie
    [basename,ext] = splitext(moviename);
    if strcmpi(ext,'.sbfmf'),
      [nr,nc,nframes,im]= sbfmf_read_header(moviename);
    else
      im = fmf_read(moviename,1,1);
    end
    % detect the arena
    [arena_center_x,arena_center_y,arena_radius] = detectarena(im);
    
  end;
elseif exist(annname,'file'),
  [arena_center_x,arena_center_y,arena_radius] = ...
    read_ann(annname,'arena_center_x','arena_center_y','arena_radius');
end

% read data
load(matname);
if exist('trx','var'),
  data = trx;
  if ~isfield(data,'off'),
    for fly = 1:length(data),
      data(fly).off = -data(fly).firstframe + 1;
      %data(fly).f2i = @(f) f - data(fly).firstframe + 1;
    end
  end
  return;
end
load(matname,'angle'); % because matlab is retarded :)
% the scipy interface has changed, everything that was once row vectors is
% now column vectors. convert to row vectors again so that my code works.
x_pos = x_pos(:)';
y_pos = y_pos(:)';
maj_ax = maj_ax(:)';
min_ax = min_ax(:)';
angle = angle(:)';
identity = identity(:)';

idscurr = unique(identity);

% normalize
if donormalize,
  x_pos = x_pos - arena_center_x; %#ok<NODEF>
  y_pos = y_pos - arena_center_y; %#ok<NODEF>
  x_pos = x_pos * 500/arena_radius;
  y_pos = y_pos * 500/arena_radius;
  maj_ax = maj_ax * 500/arena_radius; %#ok<NODEF>
  min_ax = min_ax * 500/arena_radius; %#ok<NODEF>
else
  x_pos = x_pos + 1; %#ok<NODEF>
  y_pos = y_pos + 1; %#ok<NODEF>
end

% frame number
framenumber = zeros(size(x_pos));
j = 0;
for i = 1:length(ntargets),
  framenumber(j+(1:ntargets(i))) = i;
  j = j + ntargets(i);
end;

newidentity = nan(size(identity));
for id = idscurr,
  idx = identity == id;
  datacurr.x = x_pos(idx);
  datacurr.y = y_pos(idx);
  datacurr.theta = angle(idx);
  datacurr.a = maj_ax(idx);
  datacurr.b = min_ax(idx);
  datacurr.id = id;
  datacurr.moviename = moviename;
  datacurr.firstframe = framenumber(find(idx,1));
  if exist('arena_center_x','var'),
    datacurr.arena.x = arena_center_x;
    datacurr.arena.y = arena_center_y;
    datacurr.arena.r = arena_radius;
  else
    datacurr.arena.x = nan;
    datacurr.arena.y = nan;
    datacurr.arena.r = nan;
  end
  fprintf('datacurr = \n');
  disp(datacurr);
  datacurr.off = -datacurr.firstframe + 1;
  %datacurr.f2i = @(f) f - datacurr.firstframe + 1;
  datacurr.nframes = length(datacurr.x);
  datacurr.endframe = datacurr.nframes + datacurr.firstframe - 1;
  if ~exist('data','var'),
    data = datacurr;
  else
    data(end+1) = datacurr;
  end;
  newidentity(idx) = length(data);
end;
