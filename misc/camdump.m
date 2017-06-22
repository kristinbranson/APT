function camdump(ax)
fprintf(1,'campos (%s): %s\n',ax.CameraPositionMode,mat2str(round(ax.CameraPosition)));
fprintf(1,'camtgt (%s): %s\n',ax.CameraTargetMode,mat2str(round(ax.CameraTarget)));
fprintf(1,'camang (%s): %s\n',ax.CameraViewAngleMode,mat2str(round(ax.CameraViewAngle)));
uv = ax.CameraUpVector;
if uv(3)==0
  theta = atan2(uv(2),uv(1));
  fprintf(1,'camupvec (%s): theta=%.3f deg\n',ax.CameraUpVectorMode,theta/pi*180);
else
  fprintf(1,'camupvec (%s): %s\n',ax.CameraUpVectorMode,mat2str(ax.CameraUpVector));
end
lims = axis(ax);
dx = diff(lims(1:2));
dy = diff(lims(3:4));
if numel(lims)>4
  dz = diff(lims(5:6));
else
  dz = nan;
end
fprintf(1,'axis: %s. daxis: %s\n',mat2str(round(lims)),mat2str(round([dx dy dz])));
fprintf(1,'xdir (%s): %s. ydir (%s): %s.\n',ax.XDir,ax.XDirMode,ax.YDir,ax.YDirMode);