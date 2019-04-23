function UpdateInterpColorLine(h,varargin)

[x,y,z,colors,alphas] = myparse(varargin,'x','','y','','z','','colors','','alphas','');

args = {};
if ~ischar(x),
  x = x(:);
  x = [x;flipud(x)];
  args(end+1:end+2) = {'XData',x};
end
if ~ischar(y),
  y = y(:);
  y = [y;flipud(y)];
  args(end+1:end+2) = {'YData',y};
end
if ~ischar(z),
  z = z(:);
  z = [z;flipud(z)];
  args(end+1:end+2) = {'ZData',z};
end
  
if ~ischar(colors),
  colors = [colors;flipud(colors)];
  args(end+1:end+2) = {'FaceVertexCData',colors};
end
if ~ischar(alphas),
  alphas = [alphas(:);flipud(alphas(:))];
  args(end+1:end+2) = {'FaceVertexAlphaData',alphas};
end
set(h,args{:});

