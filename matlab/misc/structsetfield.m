function s = structsetfield(s,fn,val) %#ok<INUSD>

eval(sprintf('s.%s = val;',fn));
