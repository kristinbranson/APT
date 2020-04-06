% shifts and scales input p so that output all(p >= 0) and sum(p) = 1
function p = normalizep(p)

minp = min(p);
if minp < 0,
  p = p - minp;
end;

p = p / sum(p(:));
