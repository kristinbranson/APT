function result=naturalLanguageListFromStringList(stringList)

% AL201508: Looks like dup, see civilizedStringFromCellArrayOfStrings

n=length(stringList);
if n==0
  result='';
elseif n==1
  result=stringList{1};
elseif n==2
  result=[stringList{1} ' and ' stringList{2}];
else
  for i=1:n
    if i==1
      result=[stringList{1} ',']
    elseif i<n
      result=[result ' ' stringList{i} ','];  %#ok
    elseif i==n
      result=[result ' and ' stringList{i}];  %#ok
    end
  end
end

end
