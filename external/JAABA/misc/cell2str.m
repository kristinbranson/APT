function s = cell2str(c)

s = '';

if isempty(c),
  return;
end
sc = cellfun(@cell2str_helper,c,'UniformOutput',false);
s = sprintf('%s_',sc{:});
s = s(1:end-1);

function ms = cell2str_helper(m)

if ischar(m),
  ms = m;
else
  ms = mat2str(m);
end