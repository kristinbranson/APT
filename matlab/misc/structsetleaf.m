function s = structsetleaf(s,sset,varargin)
% Set values for leaves of nested struct
% 
% s: scalar struct (typically nested)
% sset: scalar struct with fields/vals to replace
%
% structsetleaf recursively sets all leaf nodes matching fields of sset to 
% their corresponding new values. 

verbose = myparse(varargin,...
  'verbose',false ...
  );

path = '';
s = lcl(s,path,sset,verbose);

function s = lcl(s,path,sset,verbose)

fns = fieldnames(s);
for f=fns(:)',f=f{1};
  v = s.(f);
  if isstruct(v)
    s.(f) = lcl(v,[path '.' f],sset,verbose);
  elseif isfield(sset,f)
    s.(f) = sset.(f);
    if verbose
      fprintf(1,'Updated %s\n',[path '.' f]);
    end
  end
end
