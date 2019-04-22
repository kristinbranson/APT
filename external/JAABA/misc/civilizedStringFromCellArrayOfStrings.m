function str=civilizedStringFromCellArrayOfStrings(c,separator)

if ~exist('separator','var')
  separator=',';
end

n=length(c);
if n==0
  str='';
elseif n==1
  str=c{1};
elseif n==2
  str=[c{1} ' and ' c{2}];
else
  str='';
  for i=1:n-1
    str=[str c{i} separator ' '];
  end
  str=[str 'and ' c{n}];
end
  
end
