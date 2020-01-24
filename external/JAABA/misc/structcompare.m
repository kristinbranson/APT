function [issame,fns] = structcompare(s1,s2)

fns1 = fieldnames(s1);
fns2 = fieldnames(s2);

fns = union(fns1,fns2);
issame = ismember(fns,fns1) & ismember(fns,fns2);

for i = 1:numel(fns),
  
  if ~issame(i),
    fprintf('%s: not in both structs',fn);
    continue;
  end
  fn = fns{i};
  class1 = class(s1.(fn));
  class2 = class(s2.(fn));
  if ~strcmp(class1,class2),
    fprintf('%s: class mismatch\n',fn);
    issame(i) = false;
    continue;
  end
  if strcmp(class1,'char'), %#ok<ISCHR>
    if ~strcmp(s1.(fn),s2.(fn)),
      fprintf('%s: string mismatch.\n',fn);
      issame(i) = false;
    end
    continue;
  end
  
  n1 = numel(s1.(fn));
  n2 = numel(s2.(fn));
  if n1 ~= n2,
    fprintf('%s: number of elements do not match\n',fn);
    issame(i) = false;
    continue;
  end
  
  if isnumeric(s1.(fn)),
    
    nmismatch = nnz(~((isnan(s1.(fn)(:))&isnan(s2.(fn)(:))) | ...
      (s1.(fn)(:)==s2.(fn)(:))));
    if nmismatch > 0,
      fprintf('%s: %d entries do not match.\n',fn,nmismatch);
      issame(i) = false;
      continue;
    end
    
  elseif strcmp(class1,'struct'), %#ok<ISSTR>
    for j = 1:n1,
      [issamecurr] = structcompare(s1.(fn)(j),s2.(fn)(j));
      if ~all(issamecurr),
        fprintf('%s(%d): struct mismatch.\n',fn,j);
        issame(i) = false;
        break;
      end
    end
    if ~issame(i),
      break;
    end
    
  elseif strcmp(class1,'cell'), %#ok<ISCEL>
    fprintf('%s: not comparing cell entries.\n',fn);
  else
    fprintf('%s: not comparing members of class %s.\n',fn,class1);
  end

end