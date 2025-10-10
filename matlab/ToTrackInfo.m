classdef ToTrackInfo < matlab.mixin.Copyable  
  properties (Constant)
    props_numeric = {'stage','view','movieidx','frm0','frm1'};
    props_cell = {'errfile','trkfiles','moviefiles','trxfiles','trxids','croprois','calibrationfiles','calibrationdata'};
    props_array = {'trxids','croprois','frmlist'};
  end

  properties

    stages = []; % 1 x nstages
    views = []; % 1 x nviews
    nmovies = [];
    frm0 = []; % nmovies x 1
    frm1 = []; % nmovies x 1
               % if frm0 is empty, track all frames for the movies
               % if frm1 is -1, track to end of movie/trajectory
    nframestrack = [];

    frmlist = {} % nmovies x 1
    listfile = ''; % char
    movfiles = {}; % nmovies x nviews
    movidx = []; % nmovies x 1 - which movie of the full track command 
                 % this corresponds to, if tracking has been split 
                 % into multiple jobs
    trxfiles = {}; % nmovies x nviews
    trxids = {}; % nmovies x 1
                 % if trxids{i} is empty, track all animals for that movie
    croprois = {}; % nmovies x nviews
    calibrationfiles = {}; % nmovies x 1
    calibrationdata = {}; % nmovies x 1
    tblMFT = 'unset'; % table of frames to track

    trainDMC = [];  % empty or a DeepModelChainOnDisk.  Needs to be deep-copied when copying obj.
    trackid = ''; % for a particular track time
    jobid = ''; % for a particular job
    isma = false;

    % outputs
    % this will correspond to one job if these are set
    errfile = ''; % char
    logfile = ''; % char
    cmdfile = ''; % char
    killfile = ''; % char
    trackconfigfile = ''; % char
    trkfiles = {}; % nmovies x nviews x nstages

    listoutfiles = {}; % nviews . Output of list classifications
    islistjob = false; % whether we are tracking list or not
    isgtjob = false; % whether we should call gt reporting stuff at the end
    
  end

  methods
    function obj = ToTrackInfo(varargin)
      trainDMC = [];
      trkfiles = {};
      %tblMFT = [];
      for i = 1:2:numel(varargin)-1,
        prop = varargin{i};
        val = varargin{i+1};
        if strcmp(prop,'trainDMC'),
          trainDMC = val;
%         elseif strcmp(prop,'tblMFT'),
%           tblMFT = val;
        elseif strcmp(prop,'trkfiles'),
          trkfiles = val;
        else
          obj.(prop) = val;
        end
      end
      obj.checkFix();
      if ~isempty(trkfiles),
        obj.setTrkFiles(trkfiles);
      end
      if ~isempty(trainDMC),
        obj.setTrainDMC(trainDMC);
      end
      % set things that we can set automatically
      % check things that were set manually
      obj.checkFix(); 
      if obj.tblMFTIsSet,
        obj = obj.consolidateTblMFT();
      end
    end  % function
  end  % methods

  methods (Access=protected)
    function obj2 = copyElement(obj)
      % overload so that .trainDMC is deep-copied
      obj2 = copyElement@matlab.mixin.Copyable(obj);
      if ~isempty(obj.trainDMC)
        obj2.trainDMC = copy(obj.trainDMC);
      end
    end  % function
  end  % function

  methods
    function convertTblMFTToContiguous(obj)

      if ~obj.tblMFTIsSet(),
        return;
      end
      [movidx1,frm0c,frm1c,trxidsc] = ToTrackInfo.tblMFT2intervals(obj.tblMFT);
      obj.setFrm0(frm0c,'movie',movidx1);
      obj.setFrm1(frm1c,'movie',movidx1);
      obj.setTrxids(trxidsc,'movie',movidx1);
      obj.clearTblMFT();

    end

    function clearTblMFT(obj)
      obj.tblMFT ='unset';
    end


    function tf = tblMFTIsSet(obj)
      tf = ~isequal(obj.tblMFT,'unset');
    end

    function tf = frmIsSet(obj)
      frm0isset = any(~isnan(obj.frm0));
      frm1isset = any(~isnan(obj.frm1));
      tf = frm0isset || frm1isset;
    end

    function tf = trxidsIsSet(obj)
      tf = ~isempty([obj.trxids{:}]);
    end

    function n = nviews(obj)
      n = numel(obj.views);
    end

    function n = nstages(obj)
      n = numel(obj.stages);
    end

    function autoSetNMovies(obj)
      props = {'movfiles','trkfiles','frm0','frm1','frmlist','trxfiles','trxids','croprois','calibrationfiles','calibrationdata'};
      obj.nmovies = obj.autoSetSize(obj.nmovies,props,1);
      if obj.tblMFTIsSet,
        obj.nmovies = max(obj.nmovies,ToTrackInfo.getNMoviesTblMFT(obj.tblMFT));
      end
    end

    function autoSetMovidx(obj)
      nmov = size(obj.movfiles,1);
      nadd = nmov - numel(obj.movidx);
      assert(nadd>=0);
      if nadd > 0,
        if ~isempty(obj.movidx),
          assert(isequal(obj.movidx,(1:numel(obj.movidx))'));
          obj.movidx = [obj.movidx;obj.movidx(end)+(1:nadd)'];
        else
          obj.movidx = (1:nadd)';
        end
      end
    end

    function nviews = autoSetNViews(obj)
      nviews = [];
      if ~isempty(obj.views),
        nviews = obj.nviews();
      end
      props = {'movfiles','trkfiles','trxfiles','croprois'};
      nviews = obj.autoSetSize(nviews,props,2);
      if nviews == 0,
        nviews = 1;
      end
    end

    function nstages = autoSetNStages(obj)
      nstages = [];
      if ~isempty(obj.stages),
        nstages = obj.nstages;
      end
      props = {'trkfiles'};
      nstages = obj.autoSetSize(nstages,props,3);
      if nstages == 0,
        nstages = 1;
      end
    end

    function n = autoSetSize(obj,n,props,dim)
      for propi = 1:numel(props),
        prop = props{propi};
        if ~isempty(obj.(prop)),
          if isempty(n),
            n = size(obj.(prop),dim);
          else
            assert(size(obj.(prop),dim)==n);
          end
        end
      end
      if isempty(n),
        n = 0;
      end
    end


    function checkFix(obj)

      % determine number of movies
      obj.autoSetNMovies();

      % determine number of views
      nviews = obj.autoSetNViews();
      if isempty(obj.views),
        obj.views = 1:nviews;
      else
        obj.views = obj.views(:)';
      end

      % determine number of stages
      nstages = obj.autoSetNStages();
      if isempty(obj.stages),
        obj.stages = 1:nstages;
      else
        obj.stages = obj.stages(:)';
      end

      if obj.tblMFTIsSet(),
        assert(~obj.frmIsSet && ~obj.trxidsIsSet);
      end

      if obj.frmIsSet(),
        obj.frm0 = obj.frm0(:);
        szassert(obj.frm0,[obj.nmovies,1]);
        obj.frm1 = obj.frm1(:);
        szassert(obj.frm1,[obj.nmovies,1]);
      end
      if ~isempty(obj.frmlist),
        obj.frmlist = obj.frmlist(:);
        szassert(obj.frmlist,[obj.nmovies,1]);
      end

      sz = [obj.nmovies,nviews,nstages];
      if isempty(obj.trkfiles),
        obj.trkfiles = repmat({''},sz);
      else
        szassert(obj.trkfiles,sz);
      end

      sz = [obj.nmovies,nviews];
      if isempty(obj.movfiles),
        obj.movfiles = repmat({''},sz);
      else
        szassert(obj.movfiles,sz);
      end
      obj.autoSetMovidx();
      if ~isempty(obj.trxfiles),
        szassert(obj.trxfiles,[obj.nmovies,nviews]);
      end
      if obj.trxidsIsSet(),
        obj.trxids = obj.trxids(:);
        szassert(obj.trxids,[obj.nmovies,1]);
      end
      if ~isempty(obj.croprois),
        szassert(obj.croprois,[obj.nmovies,obj.nviews]);
      end
      if ~isempty(obj.calibrationfiles),
        obj.calibrationfiles = obj.calibrationfiles(:);
        szassert(obj.calibrationfiles,[obj.nmovies,1]);
      end
      if ~isempty(obj.calibrationdata),
        obj.calibrationdata = obj.calibrationdata(:);
        szassert(obj.calibrationdata,[obj.nmovies,1]);
      end
      if obj.tblMFTIsSet,
        obj.tblMFT = MFTable.sortCanonical(obj.tblMFT);
      end

    end
    function setTblMFT(obj,tblMFT1)
      obj.tblMFT = MFTable.sortCanonical(tblMFT1);
    end
    function obj = consolidateTblMFT(obj)
      if ~obj.tblMFTIsSet,
        return;
      end
      [movskeep] = unique(obj.tblMFT.mov);
      if numel(movskeep) < obj.nmovies,
        obj = obj.selectSubset('movie',movskeep);
      end
      
    end

    function idx = select(obj,prop,varargin)
      if numel(varargin) == 1,
        idx = varargin{1};
        return;
      end
      ndim = ToTrackInfo.getNdim(prop);
      nviews = obj.nviews;
      nstages = obj.nstages;
      isunset = isempty(obj.(prop)) && obj.nmovies > 0;
      if isunset,
        sz = [obj.nmovies,obj.nviews,obj.nstages];
        sz = sz(1:ndim);
        if ndim == 1,
          sz = [sz,1];
        end
      else
        sz = size(obj.(prop));
        if numel(sz) < ndim,
          sz = [sz,ones(1,ndim-numel(sz))];
        end
      end

      idx = true(sz);
      assert(obj.nmovies == sz(1));
      if ndim >= 2,
        assert(nviews == sz(2));
      end
      if ndim >= 3,
        assert(nstages == sz(3));
      end
      assert(ndim <= 3);
      for i = 1:2:numel(varargin)-1,
        switch varargin{i}
          case 'movie',
            midx = (1:obj.nmovies)';
            idx = idx & ismember(midx,varargin{i+1});
          case 'view',
            if ndim >= 2,
              views1 = reshape(obj.views,[1,nviews,1]);
              idx = idx & ismember(views1,varargin{i+1});
            end
          case 'stage',
            if ndim >= 3,
              stages1 = reshape(obj.stages,[1,1,nstages]);
              idx = idx & ismember(stages1,varargin{i+1});
            end
          otherwise
            error('not implemented: %s',varargin{i});
        end
        
      end
    end
    
    function v = getFrm0(obj,varargin)
      if isempty(varargin) || isempty(obj.frm0),
        v = obj.frm0;
        return;
      end
      idx = obj.select('frm0',varargin{:});
      v = obj.frm0(idx);
    end
    function setFrm0(obj,v,varargin)
      v = round(v);
      v(isnan(v)) = 1;
      if isempty(varargin),
        obj.frm0 = v;
        return;
      end
      if isempty(obj.frm0),
        obj.frm0 = nan(obj.nmovies,1);
      end
      idx = obj.select('frm0',varargin{:});
      if isempty(v),
        v = nan;
      end
      obj.frm0(idx) = v;
    end
    function v = getFrm1(obj,varargin)
      if isempty(varargin) || isempty(obj.frm1),
        v = obj.frm1;
        return;
      end
      idx = obj.select('frm1',varargin{:});
      v = obj.frm1(idx);
    end
    function setFrm1(obj,v,varargin)
      v = round(v);
      v(isnan(v)) = inf;
      if isempty(varargin),
        obj.frm1 = v;
        return;
      end
      if isempty(obj.frm1),
        obj.frm1 = nan(obj.nmovies,1);
      end
      idx = obj.select('frm1',varargin{:});
      if isempty(v),
        v = nan;
      end
      obj.frm1(idx) = v;
    end
    function v = getFrmlist(obj,varargin)
      if isempty(varargin) || isempty(obj.frmlist),
        v = obj.frmlist;
        return;
      end
      idx = obj.select('frmlist',varargin{:});
      v = obj.frmlist(idx);
    end
    function setFrmlist(obj,v,varargin)
      if isempty(varargin),
        obj.frmlist = v;
        return;
      end
      if isempty(obj.frmlist)
        if isempty(v),
          return;
        end
        obj.frmlist = cell(obj.nmovies,1);
      end
      idx = obj.select('frmlist',varargin{:});
      obj.frmlist(idx) = ToTrackInfo.setCellStrHelper(idx,v);
    end
    function v = getErrfile(obj)
      v = obj.errfile;
    end
    function setErrfile(obj,v)
      obj.errfile = v;
    end
    function v = getLogFile(obj)
      v = obj.logfile;
    end
    function setLogfile(obj,v)
      obj.logfile = v;
    end
    function v = getKillfile(obj)
      v = obj.killfile;
    end
    function setKillfile(obj,v)
      obj.killfile = v;
    end    function v = getCmdfile(obj)
      v = obj.cmdfile;
    end
    function setCmdfile(obj,v)
      obj.cmdfile = v;
    end
    function [v,idx] = getTrkFiles(obj,varargin)
      if isempty(varargin),
        v = obj.trkfiles;
        idx = 1:numel(v);
        return;
      end
      idx = obj.select('trkfiles',varargin{:});
      v = obj.trkfiles(idx);
    end

    function clearTrainDMC(obj,reset)
      obj.trainDMC = [];
      if reset,
        obj.trkfiles = {};
        obj.logfile = '';
        obj.killfile = '';
        obj.errfile = '';
        obj.cmdfile = '';
      end
    end

    function v = trkfilesUnset(obj)
      v = isempty(obj.trkfiles) || all(cellfun(@isempty,obj.trkfiles(:)));
    end

    function setDefaultFiles(obj,reset)
      if nargin < 2,
        reset = false;
      end
      if isempty(obj.trackid),
        warning('trackid not set');
        return;
      end
      if isempty(obj.trainDMC),
        warning('trainDMC not set');
        return;
      end
      if reset || obj.trkfilesUnset(),
        obj.setDefaultTrkfiles(reset);
      end
      if reset || isempty(obj.trackconfigfile),
        obj.setDefaultTrackConfigFile();
      end

      % rest of files require jobid to be set
      if isempty(obj.jobid),
        %warning('jobid not set');
        return;
      end
      if reset || isempty(obj.logfile),
        obj.setDefaultLogfile();
      end
      if reset || isempty(obj.killfile),
        obj.setDefaultKillfile();
      end
      if reset || isempty(obj.errfile),
        obj.setDefaultErrfile();
      end
      if reset || isempty(obj.cmdfile),
        obj.setDefaultCmdfile();
      end
    end

    function setTrainDMC(obj,v)
      obj.trainDMC = v;
    end

    function setJobid(obj,v)
      obj.jobid = v;
    end
    function v = getJobid(obj)
      v = obj.jobid;
    end

    function setTrackid(obj,v)
      obj.trackid = v;
    end
    function v = getTrackid(obj)
      v = obj.trackid;
    end


    function setDefaultTrackid(obj)
      obj.trackid = datestr(now,'yyyymmddTHHMMSS');
    end

    function setDefaultJobid(obj)
      obj.jobid = 'job1';
    end

    function v = trackjobid(obj)
      v = ['track_',obj.trackid,'_',obj.jobid];
    end

   
    function [trnstrs,idx] = getTrnStrs(obj,varargin)
      [modelChainID,idx] = obj.trainDMC.getModelChainID(varargin{:});
      iter = obj.trainDMC.getIterCurr(idx);
      vws = obj.trainDMC.getView(idx);
      trnstrs = cell(1,numel(idx));
      for i = 1:numel(idx),
        trnstrs{i} = sprintf('trn%s_view%d_iter%d',modelChainID{i},vws(i),iter(i));
      end
    end

    function setDefaultTrackConfigFile(obj)
      if isempty(obj.trkfiles) || isempty(obj.trkfiles{1}),
        warning('trkfiles must be set to set default track config file');
      end
      [p,n] = fileparts(obj.trkfiles{1});
      obj.trackconfigfile = fullfile(p,['trkconfig_',n,'.json']);
    end

    function setDefaultTrkfiles(obj,reset)
      if nargin < 2,
        reset = false;
      end
      if isempty(obj.trainDMC),
        warning('trainDMC must be set to set default trk files.')
        return;
      end
      if isempty(obj.trackid),
        warning('trackid must be set to set default trk files.');
        return;
      end

      obj.trkfiles = cell([obj.nmovies,obj.nviews,obj.nstages]);
      for ivw = 1:obj.nviews,
        view = obj.views(ivw);
        for istage = 1:obj.nstages,
          stage = obj.stages(istage);
          trkoutdir = DeepModelChainOnDisk.getCheckSingle(obj.trainDMC.dirTrkOutLnx('view',view-1,'stage',stage));
          trnstr = DeepModelChainOnDisk.getCheckSingle(obj.getTrnStrs('view',view-1,'stage',stage));
          for imov = 1:obj.nmovies,
            if ~(reset || isempty(obj.trkfiles{imov,ivw,istage})),
              continue;
            end
            mov = DeepModelChainOnDisk.getCheckSingle(obj.getMovfiles('movie',imov,'view',view,'stage',stage));
            [~,movS] = fileparts(mov);
            % add hash of movie path because sometimes all the movies have
            % the same name! Eg. Alice's projects
            shash = string2hash(mov);
            shash = shash(1:6);
            trkfilestr = [movS '_' shash '_' trnstr '_' obj.trackid '.trk'];
            obj.trkfiles{imov,ivw,istage} = [trkoutdir,'/',trkfilestr];
          end
        end
      end
    end

    function id = getId(obj)
      % i don't understand why this is so complicated -- i think we could
      % just use jobid if it is unique
      %trnstr = DeepModelChainOnDisk.getCheckSingle(obj.getTrnStrs(1));
      %mov = DeepModelChainOnDisk.getCheckSingle(obj.getMovfiles(1));
      %[~,movS] = fileparts(mov);
      %id = [movS '_' trnstr '_' obj.trackjobid];
      id = obj.trackjobid;
    end

    function f = getDefaultOutfile(obj)      
      trkoutdir1 = DeepModelChainOnDisk.getCheckSingle(obj.trainDMC.dirTrkOutLnx(1));
      id = obj.getId();
      f = [ trkoutdir1 '/' id ] ;
    end


    function setDefaultLogfile(obj)
      
      obj.logfile = [obj.getDefaultOutfile,'.log'];

    end

    function setDefaultErrfile(obj)

      obj.errfile = [obj.getDefaultOutfile,'.err'];

    end


    function setDefaultKillfile(obj)
      
      obj.killfile = [obj.getDefaultOutfile,'.KILLED'];

    end
    function setDefaultCmdfile(obj)
      obj.cmdfile = [obj.getDefaultOutfile,'.cmd'];
    end

    function setTrkFiles(obj,v,varargin)
      nstages = obj.nstages;
      nviews = obj.nviews;
      [stages1] = myparse_nocheck(varargin,'stage',obj.stages);

      if isempty(obj.trkfiles) && ~isempty(varargin),
        obj.trkfiles = repmat({''},[obj.nmovies,obj.nviews,obj.nstages]);
      end

      if numel(stages1) == 1,
        idx = obj.select('trkfiles',varargin{:});
        obj.trkfiles(idx) = ToTrackInfo.setCellStrHelper(idx,v);
      else
        if isempty(varargin),
          if ~iscell(v),
            v = {v};
          end
          if numel(v) == obj.nmovies*nviews*nstages,
            obj.trkfiles = reshape(v,[obj.nmovies,nviews,nstages]);
          else
            assert(numel(v)==obj.nmovies*nviews);
            v = repmat(reshape(v,[obj.nmovies,nviews]),[1,1,nstages]);
            v(:,:,1:end-1) = ToTrackInfo.addAutoStageNames(v(:,:,1:end-1),obj.stages(1:end-1));
            obj.trkfiles = v;
          end
        else
          idx = obj.select('trkfiles',varargin{:});
          if nnz(idx) > numel(v),
            newv = ToTrackInfo.addAutoStageNames(v,stages1);  
            assert(numel(newv) == nnz(idx));
            obj.trkfiles(idx) = newv;
          else
            obj.trkfiles(idx) = v;
          end
        end
      end
    end  % function
    
    function v = getListfile(obj)
      v = obj.listfile;
    end

    function setListfile(obj,v)
      obj.listfile = v;
    end

    function makeListFile(obj, isgt, backend)
      % Make the TrackList*.json file for tracking.  Throws if something goes wrong.
      assert(~isequal(obj.tblMFT,'unset'),'No table has been set')
      if nargin<2
        isgt = false;
      end

      trnstr = obj.trainDMC.getTrainID{1} ;
      nowstr = datestr(now(), 'yyyymmddTHHMMSS') ;
      if isgt
        extrastr = '_gt';
      else
        extrastr = '';
      end
      listfilestr = [ 'TrackList_' trnstr '_' nowstr  extrastr '.json'];
      listfile = fullfile(obj.trainDMC.rootDir,listfilestr);
      obj.setListfile(listfile);
      listoutfiles = cell(1,numel(obj.views));
      for ndx = 1:numel(obj.views)
        outlistfilestr = sprintf('preds_%s_%s%s_view%d.mat', trnstr, nowstr,extrastr,ndx);
        listoutfiles{ndx} = fullfile(obj.trainDMC.rootDir,outlistfilestr);
      end
      obj.listoutfiles = listoutfiles;

      if ~isempty(obj.trxfiles)
        args = {'trxFiles',obj.trxfiles};
      else
        args = {};
      end
      if ~isempty(obj.croprois)
        args = [args {'croprois',obj.croprois}] ;
      end
      backend.trackWriteListFile(...
        obj.movfiles, obj.movidx, obj.tblMFT, obj.listfile, args{:}) ;
      obj.islistjob = true;
    end  % backend

    function v = getMovidx(obj,varargin)
      if isempty(varargin),
        v = obj.movidx;
        return;
      end
      idx = obj.select('movidx',varargin{:});
      v = obj.movidx(idx);
    end
    function v = getMovfiles(obj,varargin)
      if isempty(varargin),
        v = obj.movfiles;
        return;
      end
      idx = obj.select('movfiles',varargin{:});
      v = obj.movfiles(idx);
    end
    function setMovfiles(obj,v,varargin)
      if isempty(varargin),
        obj.movfiles = v;
        return;
      end
      if isempty(obj.movfiles),
        obj.movfiles = repmat({''},[obj.nmovies,obj.nviews]);
      end
      idx = obj.select('movfiles',varargin{:});
      obj.movfiles(idx) = ToTrackInfo.setCellStrHelper(idx,v);
      if size(obj.movfiles,1) > numel(obj.movidx),
        obj.autoSetMovidx();
      end

    end
    function v = getTrxFiles(obj,varargin)
      if isempty(obj.trxfiles) || isempty(varargin),
        v = obj.trxfiles;
        return;
      end
      idx = obj.select('trxfiles',varargin{:});
      v = obj.trxfiles(idx);
    end
    function setTrxFiles(obj,v,varargin)
      if isempty(varargin),
        obj.trxfiles = v;
        return;
      end
      if isempty(obj.trxfiles),
        if isempty(v),
          return;
        end
        obj.trxfiles = repmat({''},[obj.nmovies,obj.nviews]);
      end
      idx = obj.select('trxfiles',varargin{:});
      if isempty(v),
        v = repmat({''},nnz(idx),1);
      end
      obj.trxfiles(idx) = ToTrackInfo.setCellStrHelper(idx,v);
    end
    function v = getTrxids(obj,varargin)
      if isempty(obj.trxids) || isempty(varargin),
        v = obj.trxids;
        return;
      end
      idx = obj.select('trxids',varargin{:});
      v = obj.trxids(idx);
    end
    function setTrxids(obj,v,varargin)
      if isempty(varargin),
        obj.trxids = v;
        return;
      end
      if isempty(obj.trxids),
        obj.trxids = cell(obj.nmovies,1);
      end
      idx = obj.select('trxids',varargin{:});
      n = nnz(idx);
      if isnumeric(v),
        v = repmat({v},[n,1]);
      elseif isempty(v),
        v = repmat({[]},[n,1]);
      elseif numel(v) == 1,
        v = repmat(v,[n,1]);
      end
      assert(numel(v)==n);
      obj.trxids(idx) = v;
    end
    function v = getCroprois(obj,varargin)
      if isempty(varargin) || isempty(obj.croprois),
        v = obj.croprois;
        return;
      end
      if isempty(obj.croprois),
        obj.croprois = cell([obj.nmovies,obj.nviews]);
      end
      idx = obj.select('croprois',varargin{:});
      v = obj.croprois(idx);
    end
    function setCroprois(obj,v,varargin)
      if isempty(varargin),
        obj.croprois = v;
        return;
      end
      if isempty(obj.croprois),
        obj.croprois = cell(obj.nmovies,obj.nviews);
      end
      idx = obj.select('croprois',varargin{:});
      n = nnz(idx);
      if isnumeric(v),
        assert(numel(v)==4);
        v = repmat({v},[n,1]);
      elseif isempty(v),
        v = repmat({[]},[n,1]);
      elseif numel(v) == 1,
        v = repmat(v,[n,1]);
      end
      assert(numel(v)==n);
      obj.croprois(idx) = v;
    end
    function v = getCalibrationfiles(obj,varargin)
      if isempty(obj.calibrationfiles) || isempty(varargin),
        v = obj.calibrationfiles;
        return;
      end
      idx = obj.select('calibrationfiles',varargin{:});
      v = obj.calibrationfiles(idx);
    end
    function setCalibrationfiles(obj,v,varargin)
      if isempty(varargin),
        obj.calibrationfiles = v;
        return;
      end
      if isempty(obj.calibrationfiles),
        obj.calibrationfiles = repmat({''},[obj.nmovies,1]);
      end
      idx = obj.select('calibrationfiles',varargin{:});
      if isempty(v),
        v = repmat({''},[nnz(idx),1]);
      end
      obj.calibrationfiles(idx) = ToTrackInfo.setCellStrHelper(idx,v);
    end
    function v = getCalibrationdata(obj,varargin)
      if isempty(obj.calibrationdata) || isempty(varargin),
        v = obj.calibrationdata;
        return;
      end
      idx = obj.select('calibrationdata',varargin{:});
      v = obj.calibrationdata(idx);
    end
    function setCalibrationdata(obj,v,varargin)
      if isempty(varargin),
        obj.calibrationdata= v;
        return;
      end
      if isempty(obj.calibrationdata),
        obj.calibrationdata = repmat({[]},[obj.nmovies,1]);
      end
      idx = obj.select('calibrationdata',varargin{:});
      if isempty(v),
        return;
      end

      n = nnz(idx);
      if ~iscell(v),
        v = repmat({v},[n,1]);
      end
      obj.calibrationdata(idx) = v;
    end
    function addTblMFT(obj,tblMFTadd,movfilesnew,movidxnew)

      nmoviesnew = ToTrackInfo.getNMoviesTblMFT(tblMFTadd);
      if nmoviesnew > obj.nmovies,
        if nargin >= 3,
          movfilesadd = movfilesnew(obj.nmovies+1:end,:);
          if nargin >= 4,
            movidxadd = movidxnew(obj.nmovies+1:end);
            assert(size(movfilesadd,1)==numel(movidxadd));
          else
            movidxadd = obj.nmovies+(1:numel(movfilesadd))';
          end
        else
          movfilesadd = {};
          movidxadd = [];
        end
        obj.addMovies('nmovies',nmoviesnew-obj.nmovies,'movfiles',movfilesadd,'movidx',movidxadd);
      end
      if ~obj.tblMFTIsSet,
        obj.setTblMFT(tblMFTadd);
      else
        tblMFTadd = tblMFTadd(:,obj.tblMFT.Properties.VariableNames);
        ism = tblismember(tblMFTadd,obj.tblMFT,MFTable.FLDSID);
        obj.setTblMFT(tblvertcatsafe(obj.tblMFT,tblMFTadd(~ism,:)));
      end

    end

    function tf = isempty(obj)
      if obj.nmovies == 0,
        tf = true;
        return;
      end
      if obj.tblMFTIsSet,
        if ~isempty(obj.tblMFT),
          tf = false;
          return;
        end
      end
      if ~obj.frmIsSet,
        tf = false;
        return;
      end
      if any(obj.frm1 >= obj.frm0),
        tf = false;
        return;
      end
    end

    function removeTblMFT(obj,tblMFTremove)
      if obj.tblMFTIsSet,
        obj.setTblMFT(MFTable.tblDiff(obj.tblMFT,tblMFTremove));
        return;
      end
      error('removing tblMFT from frm interval specification not implemented');
    end

    function addMovies(obj,varargin)

      [nmoviesadd,movfilesadd,frm0add,frm1add,frmlistadd,...
        trkfilesadd,trxfilesadd,trxidsadd,croproiadd,...
        calibrationfilesadd,calibrationdataadd,...
        tblMFTadd,movidxadd] = ...
        myparse(varargin,...
        'nmovies',[],'movfiles',{},...
        'frm0',[],'frm1',[],'frmlistadd',{},...
        'trkfiles',{},'trxfiles',{},...
        'trxids',{},'croprois',{},...
        'calibrationfiles',{},'calibrationdata',{},...
        'tblMFTadd',[],...
        'movidx',[]);

      vs = {movfilesadd,trkfilesadd,frm0add,frm1add,frmlistadd,trxfilesadd,trxidsadd,croproiadd,calibrationfilesadd,calibrationdataadd,movidxadd};
      nmoviesadd = ToTrackInfo.autoSetSizeHelper(nmoviesadd,vs,1);
      if ~isempty(tblMFTadd),
        nmoviesadd = max(nmoviesadd,ToTrackInfo.getNMoviesTBLMFT(tblMFTadd));
      end
      if nmoviesadd == 0,
        return;
      end
      nviews = obj.nviews;
      nstages = obj.nstages;
      
      if isempty(movfilesadd),
        movfilesadd = repmat({''},[nmoviesadd,nviews]);
      end
      obj.movfiles(end+1:end+nmoviesadd,:) = movfilesadd;
      if isempty(movidxadd),
        movidxadd = max(obj.getMovidx())+(1:nmoviesadd)';
      else
        assert(isempty(intersect(movidxadd,obj.getMovidx)));
      end
      obj.movidx(end+1:end+nmoviesadd) = movidxadd;
      if ~isempty(tblMFTadd) && istable(tblMFTadd),
        tblMFTadd.mov = tblMFTadd.mov + obj.nmovies;
        obj.setTblMFT(cat(1,obj.tblMFT,tblMFTadd));
      end

      if ~isempty(obj.frm0) || ~isempty(frm0add),
        if isempty(obj.frm0),
          obj.frm0 = ones([obj.nmovies,1]);
        end
        if isempty(frm0add),
          frm0add = ones([nmoviesadd,1]);
        end
        obj.frm0(end+1:end+nmoviesadd) = frm0add;
      end

      if ~isempty(obj.frm1) || ~isempty(frm1add),
        if isempty(obj.frm1),
          obj.frm1 = inf([obj.nmovies,1]);
        end
        if isempty(frm1add),
          frm1add = inf([nmoviesadd,1]);
        end
        obj.frm1(end+1:end+nmoviesadd) = frm1add;
      end

      if ~isempty(obj.frmlist) || ~isempty(frmlistadd),
        if isempty(obj.frmlist),
          obj.frmlist = cell([obj.nmovies,1]);
        end
        if isempty(frmlistadd),
          frmlistadd = cell([nmoviesadd,1]);
        end
        obj.frmlist(end+1:end+nmoviesadd) = frmlistadd;
      end


      if isempty(trkfilesadd),
        trkfilesadd = repmat({''},[nmoviesadd,nviews,nstages]);
      end
      obj.trkfiles(end+1:end+nmoviesadd,:,:) = trkfilesadd;

      if ~isempty(obj.trxfiles) || ~isempty(trxfilesadd),
        if isempty(obj.trxfiles),
          obj.trxfiles = repmat({''},[obj.nmovies,nviews]);
        end
        if isempty(trxfilesadd),
          trxfilesadd = repmat({''},[nmoviesadd,nviews]);
        end
        obj.trxfiles(end+1:end+nmoviesadd,:) = trxfilesadd;
      end

      if ~isempty(obj.trxids) || ~isempty(trxidsadd),
        if isempty(obj.trxids),
          obj.trxids = repmat({[]},[obj.nmovies,1]);
        end
        if isempty(trxidsadd),
          trxidsadd = repmat({[]},[nmoviesadd,1]);
        end
        obj.trxids(end+1:end+nmoviesadd) = trxidsadd;
      end
      
      if ~isempty(obj.calibrationfiles) || ~isempty(calibrationfilesadd),
        if isempty(obj.calibrationfiles),
          obj.calibrationfiles = repmat({''},[obj.nmovies,1]);
        end
        if isempty(calibrationfilesadd),
          calibrationfilesadd = repmat({''},[nmoviesadd,1]);
        end
        obj.calibrationfiles(end+1:end+nmoviesadd,:) = calibrationfilesadd;
      end

      if ~isempty(obj.calibrationdata) || ~isempty(calibrationdataadd),
        if isempty(obj.calibrationdata),
          obj.calibrationdata = repmat({[]},[obj.nmovies,1]);
        end
        if isempty(calibrationdataadd),
          calibrationdataadd = repmat({[]},[nmoviesadd,1]);
        end
        obj.calibrationdata(end+1:end+nmoviesadd,:) = calibrationdataadd;
      end

      obj.nmovies = obj.nmovies + nmoviesadd;
      obj.checkFix();

    end

    function addViews(obj,varargin)

      [viewsadd,movfilesadd,...
        trkfilesadd,trxfilesadd] = ...
        myparse(varargin,...
        'viewsadd',[],'movfiles',{},...
        'trkfiles',{},'trxfiles',{});

      if isempty(viewsadd),
        return;
      end
      assert(all(~ismember(obj.views,viewsadd)));
      
      nstages = obj.nstages;
      nviewsadd = numel(viewsadd);
      
      if isempty(movfilesadd),
        movfilesadd = repmat({''},[obj.nmovies,nviewsadd]);
      end
      obj.movfiles(:,end+1:end+nviewsadd) = movfilesadd;

      if isempty(trkfilesadd),
        trkfilesadd = repmat({''},[obj.nmovies,nviewsadd,nstages]);
      end
      obj.trkfiles(:,end+1:end+nviewsadd,:) = trkfilesadd;

      if ~isempty(obj.trxfiles) || ~isempty(trxfilesadd),
        if isempty(obj.trxfiles),
          obj.trxfiles = repmat({''},[obj.nmovies,nviewsadd]);
        end
        if isempty(trxfilesadd),
          trxfilesadd = repmat({''},[obj.nmovies,nviewsadd]);
        end
        obj.trxfiles(:,end+1:end+nviewsadd) = trxfilesadd;
      end

      obj.views = [obj.views,viewsadd];

      obj.checkFix();

    end

    function addStages(obj,varargin)

      [stagesadd,trkfilesadd] = ...
        myparse(varargin,...
        'stagesadd',[],'trkfiles',{});

      if isempty(stagesadd),
        return;
      end
      assert(all(~ismember(obj.stages,stagesadd)));
      
      nviews = obj.nviews;
      nstagesadd = numel(stagesadd);
      
      if isempty(trkfilesadd),
        trkfilesadd = repmat({''},[obj.nmovies,nviews,nstagesadd]);
      end
      obj.trkfiles(:,end+1:end+nstagesadd,:) = trkfilesadd;

      obj.stages = [obj.stages,stagesadd];

      obj.checkFix();

    end
    
    function tti = selectSubset(obj,varargin)

      [movieidx1,views1,stages1] = myparse(varargin,'movie',1:obj.nmovies,...
        'view',obj.views,'stage',obj.stages);
      getargs = {'movie',movieidx1,'view',views1,'stage',stages1};
      setargs = {'movie',1:numel(movieidx1),'view',views1,'stage',stages1};
      tti = ToTrackInfo('nmovies',numel(movieidx1),'views',views1,'stages',stages1);
      tti.setTrainDMC(obj.trainDMC.copy());
      tti.setFrm0(obj.getFrm0(getargs{:}),setargs{:});
      tti.setFrm1(obj.getFrm1(getargs{:}),setargs{:});
      tti.setFrmlist(obj.getFrmlist(getargs{:}),setargs{:});
      tti.setErrfile(obj.getErrfile());
      tti.setLogfile(obj.getLogFile());
      tti.setKillfile(obj.getKillfile());
      tti.setCmdfile(obj.getCmdfile());
      tti.setTrkFiles(obj.getTrkFiles(getargs{:}),setargs{:});
      tti.setListfile(obj.getListfile());
      tti.setMovfiles(obj.getMovfiles(getargs{:}),setargs{:});
      tti.movidx = obj.getMovidx(getargs{:});
      tti.setTrxFiles(obj.getTrxFiles(getargs{:}),setargs{:});
      tti.setTrxids(obj.getTrxids(getargs{:}),setargs{:});
      tti.setCroprois(obj.getCroprois(getargs{:}),setargs{:});
      tti.setCalibrationfiles(obj.getCalibrationfiles(getargs{:}),setargs{:});
      tti.setCalibrationdata(obj.getCalibrationdata(getargs{:}),setargs{:});
      tti.setTrackid(obj.getTrackid());
      if obj.tblMFTIsSet(),
        [ism,idx] = ismember(obj.tblMFT.mov,movieidx1);
        newtbl = obj.tblMFT(ism,:);
        newtbl.mov(ism) = idx(ism);
        tti.setTblMFT(newtbl);
      end
      tti.checkFix();

    end

    function merge(obj,tti)

      % can either add views OR stages OR movies
      viewsadd = setdiff(tti.views,obj.views);
      stagesadd = setdiff(tti.stages,obj.stages);
      assert(isempty(viewsadd) || isempty(stagesadd));

      if isempty(viewsadd) && isempty(stagesadd),
        nmoviesadd = tti.nmovies;
        if isempty(intersect(obj.movidx,tti.movidx)),
          movidxadd = tti.movidx;
        else
          movidxadd = [];
        end
        obj.addMovies('nmovies',nmoviesadd,...
          'movfiles',tti.getMovfiles,...
          'frm0',tti.getFrm0,'frm1',tti.getFrm1,...
          'frmlist',tti.getFrmlist,...
          'trkfiles',tti.getTrkFiles,'trxfiles',tti.getTrxFiles,...
          'trxids',tti.getTrxids,'croprois',tti.getCroprois,...
          'calibrationfiles',tti.getCalibrationfiles,...
          'calibrationdata',tti.getCalibrationdata,...
          'tblMFT',tti.tblMFT,...
          'movidx',movidxadd);
      else
        assert(tti.nmovies == obj.nmovies);
        % does not check that things match for views that overlap
        if ~isempty(viewsadd),
          assert(isequal(obj.stages,tti.stages));
          obj.addViews('views',viewsadd,...
            'movfiles',tti.getMovfiles('view',viewsadd),...
            'trkfiles',tti.getTrkFiles('view',viewsadd),...
            'trxfiles',tti.getTrxFiles('view',viewsadd));
        else
          assert(isequal(obj.views,tti.views));
          obj.addStages('stages',stagesadd,...
            'trkfiles',tti.getTrkFiles('stage',stagesadd));
        end
      end

    end

%     function nframes = getNFrames(obj,lObj)
% 
%       [movidx,frm0,frm1,trxids,nextra] = obj.getIntervals();
%       lastframeidx = find(frm1 < 0);
%       if any(lastframe),
%         [movidx1,~,idx] = unique(movidx(lastframeidx));
%         for i = 1:numel(movidx1),
%           frm1(lastframeidx(idx==i)) = lObj.getNFramesMovFile(
%         end
%         f1(imovset) = lObj.getNFramesMovFile(obj.movfileLcl{imovset});
% 
%       end
% 
%     end

    function [movidx,frm0,frm1,trxids,nextra] = getIntervals(obj)
      if obj.tblMFTIsSet(),
        [movidx,frm0,frm1,trxids,nextra] = ToTrackInfo.tblMFT2intervals(obj.tblMFT);
      else
        movidx = 1:obj.nmovies;
        frm0 = obj.frm0;
        if isempty(frm0)
          frm0 = ones(obj.nmovies,1);
        end
        frm1 = obj.frm1;
        if isempty(frm1),
          frm1 = -ones(obj.nmovies,1);
        end
        trxids = obj.trxids;
        if isempty(trxids),
          trxids = repmat({[]},[obj.nmovies,1]);
        end
        nextra = zeros(obj.nmovies,1);

      end

      frm0(isnan(frm0)) = 1;
      frm1(isnan(frm1)|isinf(frm1)) = -1;
      frm0 = round(frm0);
      frm1 = round(frm1);

    end

    function v = propSet(obj,prop)
      v = ~isempty(obj.(prop)) && ~all(cellfun(@isempty,obj.(prop)(:)));
    end
    function v = hasTrxfiles(obj)
      v = obj.propSet('trxfiles');
    end

    function v = hasCroprois(obj)
      if iscell(obj.croprois)
        if ~isempty(obj.croprois)
        % MK: 20230307 For ma projects, this is a cell. Not tested for single 
        % animal project because I didn't have one at hand. Use 
        % /groups/branson/home/kabram/APT_projects/unmarkedMice_round7_trained.lbl
        % for testing MA.
          croprois = obj.croprois{1};
        else
          croprois = [];
        end
      else
        croprois = obj.croprois;
      end
      v = obj.propSet('croprois') && ~all(any(isnan(croprois),2),1);
    end

    function v = hasCalibrationfiles(obj)
      v = obj.propSet('calibrationfiles');
    end

    function v = hasCalibrationdata(obj)
      v = obj.propSet('calibrationdata');
    end

    function v = hasTrxids(obj)
      v = obj.propSet('trxids');
    end

    function v = containerName(obj)
      v = obj.trackjobid;
    end

    function v = trkoutdir(obj,varargin)
      [view,stage] = myparse(varargin,'view',obj.views,'stage',obj.stages);
      v = obj.trainDMC.dirTrkOutLnx('view',view-1,'stage',stage);
    end

    function [v,idx] = getPartTrkFiles(obj,varargin)
      [trkfs,idx] = obj.getTrkFiles(varargin{:});
      v = cellfun(@(x) [x,'.part'],trkfs,'Uni',0);
    end

    function v = getListOutfiles(obj,varargin)
      v = obj.listoutfiles;
    end

    function nframestrack = getNFramesTrack(obj,lObj)
      if obj.islistjob
        nframestrack = obj.getNFramesTrackList();
      else
        nframestrack = obj.getNFramesTrackMovie(lObj);
      end
    end

    function nframestrack = getNFramesTrackMovie(obj,lObj)

      % nmovies x 1 - should be the same across views and stages
      nframestrack = zeros(obj.nmovies,1);

      [movidx1,f0,f1,trxids1] = obj.getIntervals();
      idxlast = movidx1(f1 < 0);
      for movi = idxlast(:)',
        movfile = obj.getMovfiles('movie',movi,'view',1);
        f1(movi) = lObj.getNFramesMovFile(movfile{1});
      end

      if ~lObj.hasTrx,
        nframestrack = f1-f0+1;
        return;
      end
         
      for i = 1:numel(movidx1),
        movi = movidx1(i);
        trxfile = DeepModelChainOnDisk.getCheckSingle(obj.getTrxFiles('movie',movi,'view',1));
        trxinfo = lObj.GetTrxInfo(trxfile);
        ids = trxids1{i};
        if isempty(ids),
          ids = 1:trxinfo.ntgts;
        end
        ffs = max(f0(i),trxinfo.firstframes(ids));
        efs = min(f1(i),trxinfo.endframes(ids));
        nframestrack(movi) = sum(efs(ffs<=efs)-ffs(ffs<=efs)+1);
      end

    end

    function nframestrack = getNFramesTrackList(obj) 
      nframestrack = size(obj.tblMFT,1);
    end

    % function changePathsToLocalFromRemote(obj, localCacheRoot, backend)
    %   % Converts all paths in obj from paths on the backend's remote filesytem 
    %   % to their corresponding local paths.  If backend is a local-filesystem
    %   % backend, do nothing.
    % 
    %   % If backend has local filesystem, do nothing
    %   if backend.isFilesystemLocal() ,
    %     return
    %   end
    % 
    %   % Generate all the relocated paths
    %   remoteCacheRoot = backend.remoteDMCRootDir ;
    %   newmovfiles = cellfun(@(old_path)(backend.getLocalMoviePathFromRemote(old_path)), ...
    %                         obj.movfiles, ...
    %                         'UniformOutput', false) ;
    %   newtrkfiles = replace_prefix_path(obj.trkfiles, remoteCacheRoot, localCacheRoot) ;
    %   newerrfile = replace_prefix_path(obj.errfile, remoteCacheRoot, localCacheRoot) ;
    %   newlogfile = replace_prefix_path(obj.logfile, remoteCacheRoot, localCacheRoot) ;
    %   newcmdfile = replace_prefix_path(obj.cmdfile, remoteCacheRoot, localCacheRoot) ;
    %   newkillfile = replace_prefix_path(obj.killfile, remoteCacheRoot, localCacheRoot) ;
    %   newtrackconfigfile = replace_prefix_path(obj.trackconfigfile, remoteCacheRoot, localCacheRoot) ;
    %   % I was concerned that some or all of obj.calibrationfiles, obj.trxfiles, and/or obj.listoutfiles
    %   % would need to be relocated, but so far hasn't been an issue 
    %   % -- ALT, 2024-07-31
    % 
    %   % Actually write all the new paths to the obj only after all the above things
    %   % have finished, to make a borked state less likely.
    %   obj.movfiles = newmovfiles ;
    %   obj.trkfiles = newtrkfiles ;
    %   obj.errfile = newerrfile ;
    %   obj.logfile = newlogfile ;
    %   obj.cmdfile = newcmdfile ;
    %   obj.killfile = newkillfile ;
    %   obj.trackconfigfile = newtrackconfigfile ;
    % end  % function

    % function changePathsToRemoteFromWsl(obj, wslCacheRoot, backend)
    %   % Converts all paths in obj from WSL paths on the frontend's filesytem to
    %   % their corresponding paths on the backend.  If backend is a local-filesystem
    %   % backend, do nothing.
    % 
    %   % If backend has local filesystem, do nothing
    %   if backend.isFilesystemLocal() ,
    %     return
    %   end
    % 
    %   % Generate all the relocated paths
    %   remoteCacheRoot = backend.remoteDMCRootDir ;
    %   newmovfiles = cellfun(@(old_path)(backend.remote_movie_path_from_wsl(old_path)), ...
    %                         obj.movfiles, ...
    %                         'UniformOutput', false) ;
    %   newtrkfiles = linux_replace_prefix_path(obj.trkfiles, wslCacheRoot, remoteCacheRoot) ;
    %   newerrfile = linux_replace_prefix_path(obj.errfile, wslCacheRoot, remoteCacheRoot) ;
    %   newlogfile = linux_replace_prefix_path(obj.logfile, wslCacheRoot, remoteCacheRoot) ;
    %   newcmdfile = linux_replace_prefix_path(obj.cmdfile, wslCacheRoot, remoteCacheRoot) ;
    %   newkillfile = linux_replace_prefix_path(obj.killfile, wslCacheRoot, remoteCacheRoot) ;
    %   newtrackconfigfile = linux_replace_prefix_path(obj.trackconfigfile, wslCacheRoot, remoteCacheRoot) ;
    %   % I was concerned that some or all of obj.calibrationfiles, obj.trxfiles, and/or obj.listoutfiles
    %   % would need to be relocated, but so far hasn't been an issue 
    %   % -- ALT, 2024-07-31
    % 
    %   % Actually write all the new paths to the obj only after all the above things
    %   % have finished, to make a borked state less likely.
    %   obj.movfiles = newmovfiles ;
    %   obj.trkfiles = newtrkfiles ;
    %   obj.errfile = newerrfile ;
    %   obj.logfile = newlogfile ;
    %   obj.cmdfile = newcmdfile ;
    %   obj.killfile = newkillfile ;
    %   obj.trackconfigfile = newtrackconfigfile ;
    % end  % function    
  end  % methods

  methods (Static)
    
    function newv = addAutoStageNames(v,stages)

      sz = size(v);
      n = prod(sz);
      nstages = numel(stages);
      newsz = [sz,nstages];

      newv = cell([n,nstages]);
      for i = 1:n,
        [p,name,ext] = fileparts(v{i});
        for istage = 1:nstages,
          stage = stages(istage);
          newv{i,istage} = fullfile(p,sprintf('%s_stg%d%s',name,stage,ext));
        end
      end
      newv = reshape(newv,newsz);

    end

    function n = autoSetSizeHelper(n,vs,dim)
      for i = 1:numel(vs),
        v = vs{i};
        if ~isempty(v),
          if isempty(n),
            n = size(v,dim);
          else
            assert(size(v,dim)==n);
          end
        end
      end
      if isempty(n),
        n = 0;
      end
    end

    function v = setCellStrHelper(idx,v)
      n = nnz(idx);

      if ischar(v),
        v = repmat({v},[n,1]);
      elseif numel(v) == 1,
        if ~iscell(v),
          v = {v};
        end
        v = repmat(v,[n,1]);
      end      
    end

    function v = getNMoviesTblMFT(tblMFT)
      v = double(max(tblMFT.mov));
    end

    function [movidx,frm0,frm1,trxids,nextra] = tblMFT2intervals(tblMFT)

      movidx = unique(double(tblMFT.mov));
      nmov = numel(movidx);
      trxids = cell(nmov,1);
      frm0 = nan(nmov,1);
      frm1 = nan(nmov,1);
      nextra = zeros(nmov,1);
      istarget = ~MFTable.isTgtUnset(tblMFT);
      for i = 1:nmov,
        mov = movidx(i);
        idx = tblMFT.mov==mov;
        frm0(i) = min(tblMFT.frm(idx));
        frm1(i) = max(tblMFT.frm(idx));
        trxids{i} = unique(tblMFT.iTgt(idx&istarget));
        if isempty(trxids{i}),
          ntgts = 1;
        else
          ntgts = numel(trxids{i});
        end
        nextra(i) = (frm1(i)-frm0(i)+1)*ntgts - nnz(idx);
      end

    end

    function ndim = getNdim(prop)

      switch prop,
        case {'frm0','frm1','frmlist','trxids','calibrationfiles','calibrationdata','movidx','listoutfiles'},
          ndim = 1;
        case {'movfiles','trxfiles','croprois'}
          ndim = 2;
        case {'trkfiles','parttrkfiles'},
          ndim = 3;
        otherwise
          error('Unknown prop %s',prop);
      end

    end

    function vout = mergeFlatten(vin)
      if isempty(vin),
        vout = {};
        return;
      end
      vout = vin{1};
      for i = 2:numel(vin),
        vout = [vout;vin{i}(:)]; %#ok<AGROW> 
      end
    end  % function

  end  % methods (Static)
end  % classdef
