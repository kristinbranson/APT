function result = genContainerMountPathBsubDocker_(obj, tracker, cmdtype, jobinfo, varargin)
  % Return a list of paths that will need to be mounted inside the
  % Apptainer/Docker container.  Returned paths are linux-appropriate.

  % Process optional args
  [aptroot, extradirs] = ...
    myparse(varargin,...
            'aptroot',[],...
            'extra',{});
  
  assert(obj.type==DLBackEnd.Bsub || obj.type==DLBackEnd.Docker);
  
  if isempty(aptroot)
    aptroot = APT.Root;  % native path
    % switch obj.type
    %   case DLBackEnd.Bsub
    %     aptroot = obj.bsubaptroot;
    %   case DLBackEnd.Docker
    %     % could add prop to backend for this but 99% of the time for 
    %     % docker the backend should run the same code as frontend
    %     aptroot = APT.Root;  % native path
    % end
  end
  
  if ~isempty(tracker.containerBindPaths)
    assert(iscellstr(tracker.containerBindPaths),'containerBindPaths must be a cellstr.');
    fprintf('Using user-specified container bind-paths:\n');
    paths = tracker.containerBindPaths;
  elseif obj.type==DLBackEnd.Bsub && obj.jrcsimplebindpaths
    fprintf('Using JRC container bind-paths:\n');
    paths = {'/groups';'/nrs'};
  else
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
    %dlroot = [aptroot '/deepnet'];
    % AL 202108: include all of <APT> due to git describe cmd which
    % looks in <APT>/.git
    paths0 = [cacheDir;aptroot;projbps(:);extradirs(:);homeancestors(:)];
    paths = FSPath.commonbase(paths0,1);
    %paths = unique(paths);
  end
  
  cellfun(@(x)fprintf('  %s\n',x),paths);
  result = wsl_path_from_native(paths) ;
end  % function