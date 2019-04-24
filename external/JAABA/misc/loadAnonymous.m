function x=loadAnonymous(fileName)

s=load(fileName,'-mat');
fieldNames=fieldnames(s);
if isempty(fieldNames)
  error('loadAnonymous:noVariablesInFile', ...
        'loadAnonymous: No variables in .mat file');
elseif length(fieldNames)>1
  error('loadAnonymous:tooManyVariablesInFile', ...
        'loadAnonymous: More than one variable in .mat file');
else
  x=s.(fieldNames{1});
end

end
