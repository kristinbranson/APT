function [issame,fns] = structcompare(s1,s2,varargin)

[prefix,verbose] = myparse(varargin,'prefix','','verbose',false);

fns1 = fieldnames(s1);
fns2 = fieldnames(s2);

fns = union(fns1,fns2);
issame = ismember(fns,fns1) & ismember(fns,fns2);

for i = 1:numel(fns),
  
  fn = fns{i};
  if isempty(prefix),
    fp = fn;
  else
    fp = [prefix,'.',fn];
  end
  
  if ~issame(i),
    fprintf('%s: not in both structs\n',fp);
    continue;
  end
  class1 = class(s1.(fn));
  class2 = class(s2.(fn));
  if ~strcmp(class1,class2),
    if verbose,
      fprintf('%s: class mismatch (%s ~= %s)\n',fp,class1,class2);
    else
      fprintf('%s: class mismatch\n',fp);
    end
    issame(i) = false;
    continue;
  end
  if strcmp(class1,'char'), %#ok<ISCHR>
    if ~strcmp(s1.(fn),s2.(fn)),
      if verbose,
        fprintf('%s: string mismatch (%s ~= %s).\n',fp,s1.(fn),s2.(fn));
      else
        fprintf('%s: string mismatch.\n',fp);
      end
      issame(i) = false;
    end
    continue;
  end
  
  n1 = numel(s1.(fn));
  n2 = numel(s2.(fn));
  if n1 ~= n2,
    if verbose
      fprintf('%s: number of elements do not match (%d ~= %d)\n',fn,n1,n2);
    else
      fprintf('%s: number of elements do not match\n',fn);
    end
    issame(i) = false;
    continue;
  end
  
  if isnumeric(s1.(fn)),
    
    nmismatch = nnz(~((isnan(s1.(fn)(:))&isnan(s2.(fn)(:))) | ...
      (s1.(fn)(:)==s2.(fn)(:))));
    if nmismatch > 0,
      fprintf('%s: %d entries do not match.\n',fp,nmismatch);
      if verbose,
        fprintf('%s ~= %s\n',mat2str(s1.(fn)),mat2str(s2.(fn)));
      end
      issame(i) = false;
      continue;
    end
    
  elseif strcmp(class1,'struct'), %#ok<ISSTR>
    for j = 1:n1,
      [issamecurr] = structcompare(s1.(fn)(j),s2.(fn)(j),'prefix',fp,'verbose',verbose);
      if ~all(issamecurr),
        fprintf('%s(%d): struct mismatch.\n',fn,j);
        issame(i) = false;
      end
    end
    
  elseif strcmp(class1,'cell'), %#ok<ISCEL>
    fprintf('%s: not comparing cell entries.\n',fp);
  elseif strcmp(class1,'logical'), %#ok<ISLOG>
    nmismatch = nnz(s1.(fn)(:)~=s2.(fn)(:));
    if nmismatch > 0,
      fprintf('%s: %d entries do not match.\n',fp,nmismatch);
      if verbose,
        fprintf('%s ~= %s\n',mat2str(s1.(fn)),mat2str(s2.(fn)));
      end
    end
  else
    fprintf('%s: not comparing members of class %s.\n',fp,class1);
  end

end