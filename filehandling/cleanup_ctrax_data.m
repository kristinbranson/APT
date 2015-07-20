function [trx,savename,timestamps] = cleanup_ctrax_data(matname,moviename,in,ds,varargin)

if ~exist('ds','var')
  ds = '';
end
[savename,dosave,annname] = myparse(varargin,...
  'savename',strrep(matname,'.mat',['trx',ds,'.mat']),...
  'dosave',false,...
  'annname','');
if ~dosave,
  savename = -1;
end

in.x_pos = in.x_pos(:)';
in.y_pos = in.y_pos(:)';
in.maj_ax = in.maj_ax(:)';
in.min_ax = in.min_ax(:)';
in.angle = in.angle(:)';
in.identity = in.identity(:)';
if isfield(in,'timestamps'),
  in.timestamps = in.timestamps(:)';
  timestamps = in.timestamps;
else
  timestamps = [];
end

idscurr = unique(in.identity);

in.x_pos = in.x_pos + 1;
in.y_pos = in.y_pos + 1;

isconverted = isfield(in,'pxpermm') && isfield(in,'fps');

% frame number
framenumber = zeros(size(in.x_pos));
j = 0;
for i = 1:length(in.ntargets),
  framenumber(j+(1:in.ntargets(i))) = i;
  j = j + in.ntargets(i);
end;

newidentity = nan(size(in.identity));

arena.x = nan;
arena.y = nan;
arena.r = nan;
if ~isempty(annname) && exist(annname,'file'),
  [arena.x,arena.y,arena.r] = read_ann(annname,'arena_center_x','arena_center_y','arena_radius');
end

if isempty(idscurr),
  trx = struct('x',cell(1,0),'y',cell(1,0),...
    'theta',cell(1,0),'a',cell(1,0),'b',cell(1,0),...
    'id',cell(1,0),'moviename',cell(1,0),...
    'annname',cell(1,0),'firstframe',cell(1,0),...
    'arena',cell(1,0),'off',cell(1,0),...
    'nframes',cell(1,0),'endframe',cell(1,0),...
    'timestamps',cell(1,0));
else
  for id = idscurr,
    idx = in.identity == id;
    datacurr.x = in.x_pos(idx);
    datacurr.y = in.y_pos(idx);
    datacurr.theta = in.angle(idx);
    datacurr.a = in.maj_ax(idx);
    datacurr.b = in.min_ax(idx);
    datacurr.id = id;
    datacurr.moviename = moviename;
    if ~isempty(annname),
      datacurr.annname = annname;
    end
    datacurr.firstframe = framenumber(find(idx,1));
    datacurr.arena = arena;
    datacurr.off = -datacurr.firstframe + 1;
    %datacurr.f2i = @(f) f - datacurr.firstframe + 1;
    datacurr.nframes = length(datacurr.x);
    datacurr.endframe = datacurr.nframes + datacurr.firstframe - 1;
    if isfield(in,'timestamps'),
      if numel(in.timestamps) >= datacurr.endframe,
        datacurr.timestamps = in.timestamps(datacurr.firstframe:datacurr.endframe);
      else
        nadd = datacurr.nframes - numel(datacurr.timestamps);
        fps = nanmedian(diff(datacurr.timestamps));
        if datacurr.firstframe > numel(in.timestamps),
          toff = datacurr.firstframe - numel(in.timestamps);
          datacurr.timestamps = in.timestamps(end)+(toff:toff+nadd)/fps;
        else
          datacurr.timestamps = in.timestamps(datacurr.firstframe:end);
          datacurr.timestamps(end+1:end+nadd) = in.timestamps(end)+(0:nadd-1)/fps;
        end
      end
    end
    if isconverted,
      datacurr.pxpermm = in.pxpermm;
      datacurr.fps = in.fps;
      datacurr.x_mm = in.x_pos_mm(idx);
      datacurr.y_mm = in.y_pos_mm(idx);
      datacurr.a_mm = in.maj_ax_mm(idx);
      datacurr.b_mm = in.min_ax_mm(idx);
    end
    if ~exist('trx','var'),
      trx = datacurr;
    else
      trx(end+1) = datacurr; %#ok<AGROW>
    end;
    newidentity(idx) = length(trx);
  end
end

if dosave,
  save_tracks(trx,savename,'timestamps',timestamps);
end