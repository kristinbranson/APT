function res = structapply(s,fcn,varargin)
% Apply function to leaf notes of nested struct
% 
% s: scalar struct (typically nested)
% fcn: function handle with sig: res = fcn(fld,val)
%
% res: containers.map with keys as fully-quald paths and vals as res

verbose = myparse(varargin,...
  'verbose',false ...
  );

path = '';
res = containers.Map();
lcl(s,res,path,fcn,verbose);

function lcl(s,res,path,fcn,verbose)

fns = fieldnames(s);
for f=fns(:)',f=f{1};
  v = s.(f);
  paththis = [path '.' f];
  if isstruct(v)
    lcl(v,res,paththis,fcn,verbose);
  else
    resthis = fcn(f,v);    
    if verbose
      fprintf(1,'%s: res=%s\n',paththis,mat2str(resthis));
    end
    res(paththis) = resthis;
  end
end
