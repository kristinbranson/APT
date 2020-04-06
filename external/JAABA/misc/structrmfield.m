function s = structrmfield(s,flds)
% s = structrmfield(s,flds)
% rmfield with removal of nested fields allowed
%
% Elements of flds which are not present in s are silently ignored.
%
% Example:
% flds = {'fld1','fld2.subfld'};
% s = structrmfield(s,flds);

assert(isstruct(s) && isscalar(s));
if ischar(flds)
  flds = cellstr(flds);
end
assert(iscellstr(flds));

for f = flds(:)',f=f{1}; %#ok<FXSET>
  fsplit = regexp(f,'\.','split');
  if isscalar(fsplit)
    fldRm = fsplit{1};
    if isfield(s,fldRm)
      s = rmfield(s,fldRm);
    else
      %warning('structrmfield:noField','No field ''%s''.',fldRm);
    end
  else
    fldTop = fsplit{1};
    fldRemainder = f(numel(fldTop)+2:end);
    valTop = s.(fldTop);
    if ~isstruct(valTop)
      %warning('structrmfield:badField','Expected field ''%s'' to be a struct.',fldTop);
    else
      s.(fldTop) = structrmfield(valTop,fldRemainder);
    end
  end
end
  