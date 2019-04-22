function [arena_center_x,arena_center_y,arena_radius] = arena_params(annname,moviename)

if exist(annname,'file'),
  [arena_center_x,arena_center_y,arena_radius] = ...
    read_ann(annname,'arena_center_x','arena_center_y','arena_radius');
else
  arena_center_x = [];
end;
if isempty(arena_center_x)
  if exist(annname,'file'),
  [bg_algorithm, im, im1] = read_ann(annname,'bg_algorithm','background_median',...
    'background_mean');
  if strcmpi(bg_algorithm,'mean')
    im = im1;
  end
  else
    % read a frame of the movie
    [basename,ext] = splitext(moviename);
    if strcmpi(ext,'.sbfmf'),
      [nr,nc,nframes,im]= sbfmf_read_header(moviename);
    else
      im = fmf_read(moviename,1,1);
    end
  end
  %detect the arena
  [arena_center_x,arena_center_y,arena_radius] = detectarena(im);
end;
