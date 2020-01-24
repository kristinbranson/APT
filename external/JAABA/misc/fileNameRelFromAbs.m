function fileNameRel=fileNameRelFromAbs(fileNameAbs)

[~,name,ext]=fileparts_platind(fileNameAbs);
fileNameRel=[name ext];

end
