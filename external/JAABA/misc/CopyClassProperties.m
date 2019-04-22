function dataout = CopyClassProperties(datain,dataout)

if isstruct(datain),
  props = fieldnames(datain);
else
  props = properties(datain);
end
for i = 1:numel(props),
  dataout.(props{i}) = datain.(props{i});
end
