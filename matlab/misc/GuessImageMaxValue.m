function maxv = GuessImageMaxValue(im)

% max(im(:)) can return a per-channel vector for multi-channel or unusual
% codec reads (e.g. old Photron AVI formats). Force scalar before thresholding.
maxv = max(im(:));
maxv = max(maxv(:));
if isempty(maxv) || ~isfinite(maxv)
  maxv = 255;  % safe fallback for unreadable/empty frames
end
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