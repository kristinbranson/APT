classdef ToTrackInfoSet < matlab.mixin.Copyable

  properties    
    ttis = [];
  end

  methods
    function obj = ToTrackInfoSet(varargin)
      if nargin >= 1,
        obj.ttis = varargin{1};
      end
    end
  end

  methods (Access=protected)
    function obj2 = copyElement(obj)
      % overload so that .ttis is deep-copied
      obj2 = copyElement@matlab.mixin.Copyable(obj);
      if ~isempty(obj.ttis)
        obj2.ttis = copy(obj.ttis);
      end
    end  % function
  end  % function
  
  methods
    function X = mergeGet(obj,propname,varargin)

      [movidx0,views0,stages0] = myparse(varargin,'movie',[],...
        'view',[],'stage',[]);

      ndim = ToTrackInfo.getNdim(propname);
      nmovies = max(cat(1,obj.ttis.movidx));
      nviews = max(cat(2,obj.ttis.views));
      nstages = max(cat(2,obj.ttis.stages));
      sz0 = [nmovies,nviews,nstages];
      if ndim == 1,
        sz = [nmovies,1];
      else
        sz = sz0(1:ndim);
      end
      isfirst = true;


      for i = 1:numel(obj.ttis),
        movidx = obj.ttis(i).getMovidx();
        views = obj.ttis(i).views;
        stages = obj.ttis(i).stages;

        idx = {movidx,views,stages};
        idx = idx(1:ndim);            
        %assert(all(all(cellfun(@isempty,X(idx{:})))));
        switch propname,
          case 'movfiles',
            x = obj.ttis(i).getMovfiles();
          case 'trkfiles',
            x = obj.ttis(i).getTrkFiles();
          case 'parttrkfiles',
            x = obj.ttis(i).getPartTrkFiles();
          case 'croprois',
            x = obj.ttis(i).getCroprois();
          case 'calibrationfiles',
            x = obj.ttis(i).getCalibrationfiles();
          case 'calibrationdata',
            x = obj.ttis(i).getCalibrationdata();
          otherwise
            x = obj.ttis(i).(propname);
        end
        if isempty(x),
          continue;
        end
        if isfirst,
          X = repmat(x(1),sz);
          isfirst = false;
        end
        X(idx{:}) = x;
      end
      
      if isfirst,
        X = {};
        return;
      end

      if ~isempty(movidx0),
        X = reshape(X(movidx0,:),[nnz(movidx0),sz(2:end)]);
        sz = size(X);
      end
      if ~isempty(views0) && ndim > 1,
        X = reshape(X(:,views0,:),[sz(1),nnz(views0),sz(3:end)]);
        sz = size(X);
      end
      if ~isempty(stages0) && ndim > 2,
        X = reshape(X(:,:,stages0),[sz(1:2),nnz(stages0)]);
      end

    end  % function

    function files = getMovfiles(obj,varargin)
      files = obj.mergeGet('movfiles',varargin{:});
    end

    function files = getTrkFiles(obj,varargin)
      files = obj.mergeGet('trkfiles',varargin{:});
    end

    function files = getPartTrkFiles(obj,varargin)
      files = obj.mergeGet('parttrkfiles',varargin{:});
    end

    function X = getCroprois(obj,varargin)
      X = obj.mergeGet('croprois',varargin{:});
    end

    function files = getCalibrationfiles(obj,varargin)
      files = obj.mergeGet('calibrationfiles',varargin{:});
    end

    function X = getCalibrationdata(obj,varargin)
      X = obj.mergeGet('calibrationdata',varargin{:});
    end

    function logFiles = getLogFiles(obj)
      logFiles = cell(numel(obj.ttis),1);
      for i = 1:numel(obj.ttis),
        logFiles{i} = obj.ttis(i).getLogFile();
      end
    end
    function errFiles = getErrFiles(obj)
      errFiles = cell(numel(obj.ttis),1);
      for i = 1:numel(obj.ttis),
        errFiles{i} = obj.ttis(i).getErrfile();
      end
    end
    function killFiles = getKillfiles(obj)
      killFiles = cell(numel(obj.ttis),1);
      for i = 1:numel(obj.ttis),
        killFiles{i} = obj.ttis(i).getKillfile();
      end
    end
    function listoutfiles = getListOutfiles(obj)
      listoutfiles = {};
      for i = 1:numel(obj.ttis)        
        j = obj.ttis(i).getListOutfiles();
        listoutfiles = [listoutfiles j];  %#ok<AGROW>
      end

    end

    function v = views(obj)
      if numel(obj.ttis) == 0,
        v = [];
      else
        v = unique(cat(2,obj.ttis.views));
      end
    end

    function v = stages(obj)

      if numel(obj.ttis) == 0,
        v = [];
      else
        v = unique(cat(2,obj.ttis.stages));
      end

    end

    function v = nmovies(obj)
      if numel(obj.ttis) == 0,
        v = 0;
      else
        v = numel(unique(cat(2,obj.ttis.movidx)));
      end
    end

    function v = n(obj)
      v = numel(obj.ttis);
    end

    function v = islistjob(obj)
      if numel(obj.ttis) == 0
        v = false;
      else
        v = obj.ttis(1).islistjob;
      end
    end

    % function changePathsToLocalFromRemote(obj, remoteCacheRoot, localCacheRoot, backend)
    %   % Assuming all the paths are paths on a remote-filesystem backend, change them
    %   % all to their corresponding local paths.  The backend argument is used to
    %   % lookup local movie paths from their remote versions.
    %   n = numel(obj.ttis) ;
    %   for i = 1 : n ,
    %     obj.ttis(i).changePathsToLocalFromRemote(remoteCacheRoot, localCacheRoot, backend) ;
    %   end
    % end  % function
  end  % methods

end  % classdef
