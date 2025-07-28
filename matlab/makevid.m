function makevid(lobj,frms,outfile,varargin)

[framerate] = myparse(varargin,'framerate',10);

v = VideoWriter(outfile);
v.FrameRate = framerate;
v.Quality = 95;
open(v);
lobj.setFrameGUI(frms(1));
ax_h = get(lobj.gdata.images_all(1),'Parent');
x_lim = ax_h.XLim;
y_lim = ax_h.YLim;
for k = frms(:)'
  lobj.setFrameGUI(k);
  ax_h.XLim = x_lim;
  ax_h.YLim = y_lim;
  frame = getframe(ax_h);
  writeVideo(v,frame.cdata);
end

close(v);

