% [param1,param2,...] = read_ann(filename,'param1','param2',...)
%
% params:
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
% max_velocity_angle_weight, 
% fracframesisback, expbgfgmodel_filename, use_expbgfgmodel, 
% expbgfgmodel_llr_thresh, min_frac_frames_isback,
% background_center, background_dev, movie_height, movie_width
function varargout = read_ann(filename,varargin)

if nargin == 1,
  readall = true;
  params = struct;
else
  readall = false;
  varargout = cell(1,nargin-1);
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

  if readall,
    params.(param) = value;
  else
    varargout = set_output(param,value,varargin,varargout);
  end;

end;

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
maybestringparams = {'bg_type'};

isspecial = ismember(param,specialparams);
isstring = ismember(param,stringparams);
ispickled = ismember(param,pickledparams);
maybestring = ismember(param,maybestringparams);

if isspecial,
  sz = str2double(value);
  value = fread(fid,sz/8,'double');
elseif ispickled,
  sz = str2double(value);
  value = fread(fid,sz,'char');  
elseif isstring,
  % leave value as string
elseif maybestring,
  nvalue = str2double(value);
  if ~isnan(nvalue),
    value = nvalue;
  end
else
  tmp = str2double(value);
  if ~isempty(tmp),
    value = tmp;
  end;
end;

param = strrep(param,' ','_');