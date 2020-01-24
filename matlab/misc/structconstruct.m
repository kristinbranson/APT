function s = structconstruct(flds,sz)
% Create structure array with given fields and size
%
% flds: cellstr
% sz: desired size

if isempty(flds)
  s = reshape(struct([]),sz);
else
  dat = flds(:)';
  dat{2,1} = cell(sz);
  s = struct(dat{:});
end