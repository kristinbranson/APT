function s = structsetfield(s,fn,val) %#ok<INUSD>

if ~structisfield(s,fn),
  return;
end
eval(sprintf('s.%s = val;',fn));
