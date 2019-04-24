if ~exist('color'),
  color = lines(1);
end

if max(im(:)) > 1,
  color = color*255;
end

npts = length(xline);
if ndims(im) < 3,
  im = repmat(im,[1,1,3]);
end

[nr,nc,three] = size(im);
im = mat2cell(im,nr,nc,ones(1,3));

for i = 1:npts-1,
  
  dxline = xline(i+1) - xline(i);
  dyline = yline(i+1) - yline(i);
  m = dyline / dxline;
  dxline = abs(dxline); dyline = abs(dyline);
  if dxline > dyline,
    xlinecurr = linspace(xline(i),xline(i+1),dxline);
    ylinecurr = yline(i) + m*(xlinecurr-xline(i));
  else,
    ylinecurr = linspace(yline(i),yline(i+1),dyline);
    xlinecurr = xline(i) + (ylinecurr - yline(i))/m;
  end
  for rxline = 1:2, 
    if rxline == 1,
      roundxline = floor(xlinecurr);
    else,
      roundxline = ceil(xlinecurr);
    end
    dxline = xlinecurr - roundxline;
    for ryline = 1:2,
      if ryline == 1,
        roundyline = floor(ylinecurr);
      else,
        roundyline = ceil(ylinecurr);
      end
      dyline = ylinecurr - roundyline;
      w = max(0,sqrt(dxline.^2 + dyline.^2) - (sqrt(2) - 1));
      for c = 1:3,
        im{c}(sub2ind([nr,nc],roundyline,roundxline)) = im{c}(sub2ind([nr,nc],roundyline,roundxline)).*w + ...
            color(c)*(1-w);
      end
    end
  end
end

im = cell2mat(im);
