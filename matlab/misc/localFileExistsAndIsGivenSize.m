function result = localFileExistsAndIsGivenSize(file_name, sz)
  result = logical(exist(file_name,'file')) ;
  if result
    dirfile = dir(file_name);
    result = (dirfile.bytes==sz) ;
  end
end
