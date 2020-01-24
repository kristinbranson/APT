function fileNameAbs=absolutifyFileName(fileName,workingDirNameAbs)

if isabspath(fileName) ,
  fileNameAbs=fileName;
else
  fileNameAbs=fullfile(workingDirNameAbs,fileName);
end

end