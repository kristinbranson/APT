function getframe_cleanup(gfdata)

if ~ishandle(gfdata.haxes), return; end
noanimate('restore',get(gfdata.haxes,'parent'));
warning(gfdata.warnstate);