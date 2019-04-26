function im = drawlineonimage(im,x,y,color)

if ~exist('color'),
  color = lines(1);
end

if max(im(:)) > 1,
  color = color*255;
end

npts = length(x);
if ndims(im) < 3,
  im = repmat(im,[1,1,3]);
end

[nr,nc,three] = size(im);
im = mat2cell(im,nr,nc,ones(1,3));

for i = 1:npts-1,
  
  dx = x(i+1) - x(i);
  dy = y(i+1) - y(i);
  m = dy / dx;
  dx = abs(dx); dy = abs(dy);
  if dx > dy,
    xcurr = linspace(x(i),x(i+1),dx);
    ycurr = y(i) + m*(xcurr-x(i));
  else,
    ycurr = linspace(y(i),y(i+1),dy);
    xcurr = x(i) + (ycurr - y(i))/m;
  end
  for rx = 1:2, 
    if rx == 1,
      roundx = floor(xcurr);
    else,
      roundx = ceil(xcurr);
    end
    dx = xcurr - roundx;
    for ry = 1:2,
      if ry == 1,
        roundy = floor(ycurr);
      else,
        roundy = ceil(ycurr);
      end
      dy = ycurr - roundy;
      w = max(0,sqrt(dx.^2 + dy.^2) - (sqrt(2) - 1));
      for c = 1:3,
        im{c}(sub2ind([nr,nc],roundy,roundx)) = im{c}(sub2ind([nr,nc],roundy,roundx)).*w + ...
            color(c)*(1-w);
      end
    end
  end
end

im = cell2mat(im);
