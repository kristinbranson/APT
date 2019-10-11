function s = cell2spacedargs(c)

if ischar(c)
  s = c;
elseif iscell(c),
  s = sprintf('%s ',c{:});
  s = s(1:end-1);
else
  error('Only chars and cells can be input');
end