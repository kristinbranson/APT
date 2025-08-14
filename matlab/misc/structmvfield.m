function s = structmvfield(s,oldfn,newfn) 

if ~structisfield(s,oldfn),
  return;
end
eval(sprintf('s.%s = s.%s;',newfn,oldfn));
s = structrmfield(s,oldfn);
