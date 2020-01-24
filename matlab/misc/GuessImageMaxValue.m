function maxv = GuessImageMaxValue(im)

maxv = max(im(:));
if maxv < .5,
  % use observed value
elseif maxv <= 1,
  maxv = 1;
elseif maxv < 128
  % use observed value
elseif maxv <= 255,
  maxv = 255;
elseif maxv < 32768,
elseif maxv <= 65535,
  maxv = 65535;
end