function result = filesExistOnDisk(pathFromFileIndex)
  % Return a logical array, the same size as pathFromFileIndex (a cellstr),
  % indicating which paths refer to files that currently exist on disk.
  % Empty paths yield false.
  result = false(size(pathFromFileIndex)) ;
  for i = 1 : numel(pathFromFileIndex)
    thisPath = pathFromFileIndex{i} ;
    if ~isempty(thisPath) && exist(thisPath, 'file')
      result(i) = true ;
    end
  end
end
