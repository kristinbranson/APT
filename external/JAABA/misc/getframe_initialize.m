function gfdata = getframe_initialize(haxes)

inputType = getInputType(haxes);

if strcmp(inputType,'figure'),
    
  hfig = haxes;

  haxes = get(hfig,'children');
  if length(haxes) > 1,
    haxes = hfig;
    %error('Figure has multiple children, please input handle of axes instead of figure');
  end

else
  hfig = get(haxes,'parent');
end



gfdata.haxes = haxes;
gfdata.hfig = hfig;
gfdata.units = get(hfig,'Units');
gfdata.pos = get(hfig,'Position');
pixelsperinch = get(0,'screenpixelsperInch');
pos =  get(hfig,'position');
% If units are normalized, then re-scale paperposition calculation
% to use pixel values based on the screen size.
if strcmp(get(hfig, 'units'), 'normalized')
  screensize = get(0, 'screensize');
  screenwidth = screensize(3);
  screenheight = screensize(4);
  pos = [(pos(1) * screenwidth + 1) (pos(2) * screenheight + 1) (pos(3) * screenwidth) (pos(4) * screenheight)];
end
set(hfig, 'paperposition', pos./pixelsperinch);
renderer = get(hfig,'renderer');
if strcmp(renderer,'painters')
  renderer = 'zbuffer';
end
%Turn off warning in case opengl is not supported and
%hardcopy needs to use zbuffer
warnstate = warning('off');
gfdata.warnstate = warnstate;
noanimate('save',get(haxes,'parent'));
gfdata.hardcopy_args = {haxes,['-d',renderer],['-r',num2str(round(pixelsperinch))]};

% TODO: put this in cleanup
% noanimate('restore',get(haxes,'parent'));
% warning(warnstate);

function inputType = getInputType(frame)
if isscalar(frame) && ishandle(frame),
  inputType = get(frame,'type');
elseif isstruct(frame) & isfield(frame,'cdata')
  inputType = 'movie';
elseif isa(frame,'numeric')
  inputType = 'data';
else
  error('Invalid input argument.  Each frame must be a numeric matrix, a MATLAB movie structure, or a handle to a figure or axis.');
end
