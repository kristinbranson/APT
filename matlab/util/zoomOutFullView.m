function zoomOutFullView(hAx,hIm,resetCamUpVec)
if isequal(hIm,[])
  axis(hAx,'auto');
else
  xdata = hIm.XData;
  ydata = hIm.YData;
  set(hAx,...
    'XLim',[xdata(1)-0.5 xdata(end)+0.5],...
    'YLim',[ydata(1)-0.5 ydata(end)+0.5]);
end
axis(hAx,'image');
zoom(hAx,'reset');
if resetCamUpVec
  hAx.CameraUpVectorMode = 'auto';
end
hAx.CameraViewAngleMode = 'auto';
hAx.CameraPositionMode = 'auto';
hAx.CameraTargetMode = 'auto';
