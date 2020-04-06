% t = strsplit(s,delim)
function t = strsplit_jaaba(s,delim)

if ischar(delim),
  i = strfind(s,delim);
else
  i = [];
  for j = 1:length(delim),
    i = union(i,strfind(s,delim{j}));
  end
end
i = [0,i,length(s)+1];
n = length(i)-1;
t = cell(1,n);
for j = 1:n,
  t{j} = s(i(j)+1:i(j+1)-1);
end
