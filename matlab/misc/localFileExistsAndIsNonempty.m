function result = localFileExistsAndIsNonempty(file_name)
  result = logical(exist(file_name,'file')) ;
  if result
    dirfile = dir(file_name);
    result = dirfile.bytes>0;
  end
end
