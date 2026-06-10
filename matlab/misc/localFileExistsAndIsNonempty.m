function result = localFileExistsAndIsNonempty(file_name)
  % Force NFS attribute cache refresh by listing the parent directory before
  % checking the file.  Without this, exist() can return stale "not found"
  % for up to acregmax seconds (typically 30-60s) after a remote process writes
  % the file -- causing false "tracking failed" errors on bsub/cluster jobs.
  parent = fileparts(file_name) ;
  if ~isempty(parent)
    dir(parent) ;  % sends NFS READDIR, invalidates stale attribute cache
  end
  dirfile = dir(file_name) ;
  result = ~isempty(dirfile) && dirfile(1).bytes > 0 ;
end
