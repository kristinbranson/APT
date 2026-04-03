function result = genContainerMountPathBsubDocker_(obj, tracker, cmdtype, jobinfo)
  % Return a list of paths that will need to be mounted inside the
  % Apptainer/Docker container.  Returned paths are wsl-locale MetaPaths.

  assert(obj.type==DLBackEnd.Bsub || obj.type==DLBackEnd.Docker);
  
  aptRootAsChar = APT.Root;  % native path
    
  if ~isempty(tracker.containerBindPaths)
    assert(iscellstr(tracker.containerBindPaths),'containerBindPaths must be a cellstr.');
    fprintf('Using user-specified container bind-paths:\n');
    nativePathsAsChar = tracker.containerBindPaths;
    cellfun(@(path)fprintf('  %s\n',path),nativePathsAsChar);
    nativePaths = ...
      cellfun(@(pathAsChar)(apt.MetaPath(pathAsChar, apt.PathLocale.native, apt.FileRole.universal)), ...
              nativePathsAsChar,  ...
              'UniformOutput', false) ;
    result = cellfun(@(path)(path.asWsl()), nativePaths, 'UniformOutput', false) ;
    return
  end
    
  if obj.type==DLBackEnd.Bsub && obj.jrcsimplebindpaths
    fprintf('Using JRC container bind-paths:\n');
    wslPathsAsChar = {'/groups';'/nrs'};
    cellfun(@(path)fprintf('  %s\n',path),wslPathsAsChar);
    result = ...
      cellfun(@(pathAsChar)(apt.MetaPath(pathAsChar, apt.PathLocale.wsl, apt.FileRole.universal)), ...
              wslPathsAsChar,  ...
              'UniformOutput', false) ;
    return
  end
    
  lObj = tracker.lObj;
  
  %macroCell = struct2cell(lObj.projMacrosGetWithAuto());
  %cacheDir = obj.lObj.DLCacheDir;
  cacheDir = APT.getdotaptdirpath() ;  % native path
  assert(~isempty(cacheDir));
  
  if isequal(cmdtype,'train'),
    projbps = lObj.movieFilesAllFull(:);
    %mfafgt = lObj.movieFilesAllGTFull;
    if lObj.hasTrx,
      projbps = [projbps;lObj.trxFilesAllFull(:)];
      %tfafgt = lObj.trxFilesAllGTFull;
    end
  else
    projbps = jobinfo.getMovfiles();
    projbps = projbps(:);
    if lObj.hasTrx,
      trxfiles = jobinfo.getTrxFiles();
      trxfiles = trxfiles(~cellfun(@isempty,trxfiles));
      if ~isempty(trxfiles),
        projbps = [projbps;trxfiles(:)];
      end
    end
  end
  
  [projbps2,ischange] = GetLinkSources(projbps);
  projbps(end+1:end+nnz(ischange)) = projbps2(ischange);

	if obj.type==DLBackEnd.Docker
    % docker writes to ~/.cache. So we need home directory. MK
    % 20220922
    % add in home directory and their ancestors
    homedir = getuserdir() ;  % native path
    homeancestors = [{homedir},getpathancestors(homedir)];
    if isunix()
      homeancestors = setdiff(homeancestors,{'/'});
    end
  else
    homeancestors = {};
  end

  fprintf('Using auto-generated container bind-paths:\n');
  % AL 202108: include all of <APT> due to git describe cmd which
  % looks in <APT>/.git
  paths0 = [cacheDir;aptRootAsChar;projbps(:);homeancestors(:)];
  nativePathsAsChar = FSPath.commonbase(paths0,1);
  
  cellfun(@(path)fprintf('  %s\n',path),nativePathsAsChar);
  nativePaths = ...
    cellfun(@(pathAsChar)(apt.MetaPath(pathAsChar, apt.PathLocale.native, apt.FileRole.universal)), ...
            nativePathsAsChar,  ...
            'UniformOutput', false) ;
  result = cellfun(@(path)(path.asWsl()), nativePaths, 'UniformOutput', false) ;  
end  % function
