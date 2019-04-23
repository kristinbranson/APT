% s = structappend(sbase,snew)
% s = structappend(sbase,snew,dim)
function s = structappend(sbase,snew,dim)

if nargin < 3,
  if length(sbase) > 1,
    dim = argmax(size(sbase));
  elseif length(snew) > 1,
    dim = argmax(size(snew));
  else
    dim = 2;
  end
end

if isempty(sbase),
  s = snew;
else
  fnsnew = fieldnames(snew);
  fnsbase = fieldnames(sbase);
  fnsmissing = setdiff(fnsnew,fnsbase);
  for i = 1:numel(fnsmissing),
    fn = fnsmissing{i};
    sbase(1).(fn) = [];
  end
  fnsmissing = setdiff(fnsbase,fnsnew);
  for i = 1:numel(fnsmissing),
    fn = fnsmissing{i};
    snew(1).(fn) = [];
  end
  fnsbase = fieldnames(sbase); 
  fnsnew = fieldnames(snew);
  if ~all(strcmp(fnsbase,fnsnew)),
    [~,order] = ismember(fnsbase,fnsnew);
    s = cat(dim,sbase,orderfields(snew,order));
  else
    s = cat(dim,sbase,snew);
  end
end