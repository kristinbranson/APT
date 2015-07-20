% [param1,param2,...] = read_ann2(filename,'param1','param2',...)
%
% params:
%
% trx: trajectories
%
% version, bg_type, n_bg_std_thresh, n_bg_std_thresh_low, bg_std_min,
% bg_std_max n_bg_frames, min_nonarena, max_nonarena, arena_center_x,
% arena_center_y, arena_radius, do_set_circular_arena, bg_algorithm,
% background_median, bg_norm_type, background_mad, hfnorm, bg_norm_type,
% hm_cutoff, hm_boost, hm_order, maxarea, maxmajor, maxminor, maxecc,
% minarea, minmajor, minminor, minecc, meanarea, meanmajor, meanminor,
% meanecc, nframes_size, nstd_shape, max_jump, ang_dist_wt, center_dampen,
% angle_dampen, minbackthresh, maxpenaltymerge, maxareadelete,
% do_fix_split, splitdetection_length, splitdetection_cost, do_fix_merged,
% mergeddetection_length, mergeddetection_distance, do_fix_spurious,
% spuriousdetection_length, do_fix_lost, lostdetection_length, movie_name,
% start_frame, data_format, velocity_angle_weight,
% max_velocity_angle_weight
function varargout = read_ann2(filename,varargin)

i = find(strcmpi('trx_firstframe',varargin),1);
if ~isempty(i),
  trx_firstframe = varargin{i+1};
  varargin(i:i+1) = [];
else
  trx_firstframe = 1;
end
  
i = find(strcmpi('trx_endframe',varargin),1);
if ~isempty(i),
  trx_endframe = varargin{i+1};
  varargin(i:i+1) = [];
else
  trx_endframe = inf;
end

if nargin == 1,
  readall = true;
  readtrx = true;
else  
  readall = false;
  varargout = cell(1,nargin-1);
  readtrxi = strcmpi(varargin,'trx');
  readtrx = any(readtrxi);
end;

fid = fopen(filename,'rb');

while true,

  s = fgetl(fid);
  if strcmp(s,'end header') || ~ischar(s),
    break;
  end

  [param,value] = read_line(s,fid);
  if isempty(param),
    continue;
  end;

  if readtrx && strcmpi(param,'start_frame')
    startframe = value;
  end
  disp(param);
  
  if readall,
    params.(param) = value;
  else
    varargout = set_output(param,value,varargin,varargout);
  end;

end;

if readtrx,
  trx = struct('x',{},'y',{},'theta',{},'a',{},'b',{},'id',{});
  ids = [];
  f = max(startframe,trx_firstframe);
  nfields = 6;
  while true,
    if mod(f,300) == 0,
      fprintf('reading frame %d\n',f);
    end
    s = fgetl(fid);
    if ~ischar(s),
      break;
    end
    ss = sscanf(s,'%f');
    ntargets = floor(length(ss) / nfields);
    x = ss(1:nfields:end);
    y = ss(2:nfields:end);
    b = ss(3:nfields:end);
    a = ss(4:nfields:end);
    theta = ss(5:nfields:end);
    id = ss(6:nfields:end);
    for i = 1:ntargets,
      j = find(id(i)==ids);
      if isempty(j),
        ids(end+1) = id(i);
        j = length(ids);
        trx(j).firstframe = f;
      end
      trx(j).x(end+1) = x(i);
      trx(j).y(end+1) = y(i);
      trx(j).theta(end+1) = theta(i);
      trx(j).a(end+1) = a(i);
      trx(j).b(end+1) = b(i);
      trx(j).id(end+1) = id(i);
    end
    f = f+1;
    if f > trx_endframe,
      break;
    end
  end
  
  for i = 1:length(trx),
    trx(i).nframes = length(trx(i).x);
    trx(i).endframe = trx(i).firstframe + trx(i).nframes - 1;
    trx(i).off = -trx(i).firstframe + 1;
  end
  
  if readall,
    params.trx = trx;
  else
    varargout = set_output('trx',trx,varargin,varargout);
  end
end
if readall,
  varargout{1} = params;
end;

fclose(fid);

function out = set_output(param,value,in,out)

for i = 1:length(in),
  if strcmp(in{i},param),
    out{i} = value;
  end;
end;

function [param,value] = read_line(s,fid)

i = strfind(s,':');
if isempty(i),
  param = [];
  value = [];
  return;
end;
param = s(1:i-1);
value = s(i+1:end);

specialparams = {'background median','background mean',...
                 'background mad','background std',...
                 'hfnorm','fracframesisback',...
                 'background center','background dev',...
                 'isarena'};
stringparams = {'bg_algorithm','version','expbgfgmodel_filename','movie_name','data format'};
pickledparams = {'roipolygons'};

isspecial = ismember(param,specialparams);
isstring = ismember(param,stringparams);
ispickled = ismember(param,pickledparams);

if isspecial,
  sz = str2double(value);
  value = fread(fid,sz/8,'double');
elseif ispickled,
  sz = str2double(value);
  value = fread(fid,sz,'char');  
elseif isstring,
  % leave value as string
else
  tmp = str2double(value);
  if ~isempty(tmp),
    value = tmp;
  end;
end;

param = strrep(param,' ','_');