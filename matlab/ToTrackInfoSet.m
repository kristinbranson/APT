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
            x = obj.ttis(i).getTrkfiles();
          case 'parttrkfiles',
            x = obj.ttis(i).getParttrkfiles();
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

    end

    function files = getMovfiles(obj,varargin)
      files = obj.mergeGet('movfiles',varargin{:});
    end

    function files = getTrkfiles(obj,varargin)
      files = obj.mergeGet('trkfiles',varargin{:});
    end

    function files = getParttrkfiles(obj,varargin)
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

    function logFiles = getLogfiles(obj)
      logFiles = cell(numel(obj.ttis),1);
      for i = 1:numel(obj.ttis),
        logFiles{i} = obj.ttis(i).getLogfile();
      end
    end
    function errFiles = getErrfiles(obj)
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
        listoutfiles = [listoutfiles j];
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

  end


end