classdef TrackJob < handle
% TrackJob
%
% Represents a single tracking task to be executed on a DL host. A TrackJob
% represents a *single APT_interf call* but note this could involve 
% tracking multiple movies/views as supported by APT_interf.
%
% Typically we envision a 2d matrix of movies-to-be-tracked
% 
% mov1vw1 mov1vw2
% mov2vw1 mov2vw2
% ...
% movNvw1 movNvw2
%
% Currently TrackJob supports these tracking modalities
% 1. Entire single row. tfmultiview=1, .ivw=1:nTotalView
% 2. Single view in a single row. tfmultiview=0, .ivw=<scalar view idx>
% 3. Entire column. tfserialmultimov=1, tfmultiview=0, .ivw=<scalar view idx>
%      (Currently only implemented/checked for .tfexternal=true and .tfremote=false)


% Bulk Deep Tracking Impl Manifesto 20200318
%
% The MovieSet Array (MSA).
% 
% mov1vw1 mov1vw2 ...
% mov2vw1 mov2vw2
% ...
% movNvw1 movNvw2
% 
% This is a [nmovset x nview] array of movies to be tracked.
% Note the distinction between nmovset and nmov=nmovset*nview.
% 
% Associated with each movie is (optionally):
% * trxfile
% * output/trkfile
% * crop
%
% This is the most common set of movies-to-be-tracked by far, but eg a 
% non-rectangular subset is possible in principle.
% 
% Job breakup.
% 
% The MSA can be quite large eg hundreds, maybe thousands of rows. It
% needs to be split up/parallelized across DL nodes.  How this is done
% is constrained by the number of DL nodes available, eg: 
% 
% * If nnode is very large eg nnode>=nmov, then each individual movie
% can be processed on its own node.  
% * If nnodes is large eg nnode>=nmovset, then each movieset or row of
% the MSA can be processed on its own node, with views processed
% serially on the node. Alternatively, one full column/view of the MSA
% can be processed with one movie per node.  
% * If nnode is small eg nnode<nmovset, then one can imagine
% partitioning the MSA and assigning regions to each node.  
% * If nnode==1 then any processing will necessarily run serially.
% 
% The current MSA partitioning modalities available are
% listed/maintained in TrackJob.m with some further implementation
% specifics/constraints in DeepTracker. Each TrackJob represents a
% single tracking task on a single DL host.
% 
% Single-job API (APT_interf), MSA slicing.
% 
% On a single DL host, APT_interf can be run on a given chunk/region of
% the MSA. Currently APT_interf supports:
% - single row of movset array, ie one movieset across views
% - single column of movset array (eg specified single view 
%   if proj is MV)
% 
% Within these modalities, APT_interf's api supports specification of an
% arbitrary number of movies, along with their associated trxfiles,
% crops etc.
% 
% The target-frame (TF) array.
% 
% We drill down within the MSA to a single movie or element in the MSA.
% 
% For a given movie, the (target, frame) array is a 2d non-rectangular 
% array representing all available (t,f) tuples for that movie. Any given 
% movie has an arbitrary number of targets; each target is live for an 
% arbitrary number of frames.
%
% When tracking a movie one might want to track some subset of the full TF
% array which we call the TFset.
% 
% APTinterf track API, non-listfile.
% 
% Currently: f0/f1 specs are scalars, as are targets. That means that we
% enable tracking a fixed rectangular TFset across all movies (within a 
% single DL proc).
% 
% Next update: accept vectorized f0, f1, targets (last with argparse
% trick) This enables arbitrary startframe, endframe, and targets for
% each movie. This enables the TFset to vary across movies, the remaining 
% limitation being that the TFset must still be rectangular for any given 
% movie (constrained by trx availability etc).
% 
% Future: this limitation can be lifted in the future via a further 
% 'append-style' arg that follows some syntax/encoding, eg:
% APT -movs mov1 mov2 mov3 -trxs trx1 trx2 trx3 -outs out1 out out3 ...
% -mtf 1 -mtf 2 3 -mtf 3 1 10 100 =>
% * track mov1 in entirety, all tgts
% * track mov2, tgt 3, all frames
% * track mov3, tgt 1, frames [10,100] etc
% 
% this can be expanded a lot depending on effort, eg -mtf 3 1-4 10 100 
% 
% in which case probably effectively anything ie a fully arbitrary tfset is 
% for each movie is possible with effort.
%
% The MTF array.
% Connecting the MSA with the TF array is the MTF array which is a non-
% rectangular 3d array containing all available (mov,tgt,frm) triples. For
% multiview projects the movie implicitly carries a view index.
% 
% APTinterf track API, listfile (json).
% Using the -listfile argument potentially enables i) tracking a fully-
% arbitrary MTFset while ii) keeping a convenient record on disk of what 
% was run, for posterity and avoiding long cmdlines etc. However at this
% point the full functionality is theoretically available in the regular
% cmdline API of APT_interf and the listfile can be viewed as syntactic 
% sugar.



  properties
    
    tObj = [];
    backend = [];

    trnstr = {}; % [nView x nset] cellstr, first output of DeepTracker.getTrainStrModelFiles; corresponding to .ivw
    nowstr = 'UNIQUETAG';
    id = 'TAG_ANN';
    modelChainID = '';
    remoteRoot = '/home/ubuntu';
    remoteDataDirRel = 'data';
    
    modelfile = {}; % cellstr, [nView x nSet] corresponding to .ivw
    logdir = '';
    logfile = '';
    errfile = '';
    
    % AL: Ideally we would enforce these all to be [nmovsettrk x nView]
    lblfileLcl = '';
    movfileLcl = {}; % cellstr, [nView] corresponding to .ivw; or if 
                     % tfserialmultimov, [nSerialMov]
                     % SHOULD PROB BE [nmov x nView x nSet]
    trxfileLcl = {}; % unused if .tftrx is false. If .tftrx is true:
                     % if ~tfserialmultimov, cellstr [nView] corresponding to .ivw 
                     %    (note currently APT only supports single-view projs with trx or 2stg);
                     % if tfserialmultimov, [nSerialMov] corresponding to .movfileLcl
%     trxfileStg1Lcl = {}; % Only used if .tf2stg is true.
%                          % trxfiles produced by stage1 tracking
%                          % if ~tfserialmultimov, cellstr [nView] corresponding to .ivw 
%                          %    (note currently APT only supports single-view projs with 2stg);
%                          % if tfserialmultimov, [nSerialMov] corresponding to .movfileLcl          

    % nset==2 if .tf2stg is true
    trkfileLcl = {}; % cellstr, [nView x nset] corresponding to .ivw; or if tfserialmultimov, [nSerialMov x nset]
    trkoutdirLcl = {}; % cellstr, [nView x nset] corresponding to .ivw. STILL USED for errfile if .tfexternal
    parttrkfileLcl = {}; % cellstr, [nView x nset] corresponding to .ivw; or if tfserialmultimov, [nSerialMov x nset]
    trkfilestr = {}; % cellstr, [nView x nset] shortnames for trkfileLcl. UNUSED if .tfexternal
    
    rootdirLcl = '';
    listfilestr = '';
    listfileLcl = '';
    calibrationfileLcl = {}; % cellstr, [nSerialMov] % KB 20200504: added, but not used yet
    
    lblfileRem = '';
    movfileRem = {};
    trxfileRem = {};
    trxfileStg1Rem = {};
    trkfileRem = {};
    trkoutdirRem = {};
    parttrkfileRem = {};
    trkoutdirStg1Rem = {};
    parttrkfileStg1Rem = {};
    rootdirRem = '';
    listfileRem = '';
        
    tfmultiview = false;
    tfserialmultimov = false;
    tfexternal = false;
    tftrx = false;
    tf2stg = false;
    tfremote = false;    

    dmcLcl = []; % [nView x nset] dmc
    dmcRem = [];
    
    tMFTConc = [];
    trxids = {};
    frm0 = []; % scalar or (if tfserialmultimov) nSerialMov x 1 vector
    frm1 = []; % scalar or (if tfserialmultimov) nSerialMov x 1 vector
    cropRoi = []; % either [], or [nView x 4], or (if tfserialmultimov) [nSerialMov x 4]
    nframestrack = []; % after set via getNFramesTrack, [nmovsettrk]
    isContiguous = true; % whether to track a contiguous interval of frames 
                         % or to specify a table of individual frames to
                         % track
    
    nView = 0; % number of views in this job
    nSerialMov = 0; % number of movies; applicable if tfserialmov
    ivw = [];
    nTotalView = 0; % total number of views in project
    
    uploadfun = @(varargin) [];
    mkdirRemFun = @(varargin) [];
    rmRemFun = @(varargin) [];
    downloadfun = @(varargin) true;
    
    logcmd = '';
    codestr = '';
    ssfile = '';
  end
  
  properties (Dependent)
    nmovsettrk 

    trkfile
    containerName
    movfile
    %movlocal
    %movremote
    %trkfilelocal
    %trkfileremote
    %parttrkfilelocal
    %parttrkfileremote
    %trxlocal
    %trxremote
    snapshotfile        
  end
  
  methods
    
    function obj = TrackJob(tObj,backend,varargin)
      
      if nargin == 0,
        return;
      end
      
      [ivw,trxids,frm0,frm1,cropRoi,tMFTConc,...
        lblfileLcl,movfileLcl,trxfileLcl,trkfileLcl,...
        trkoutdirLcl,rootdirLcl,...
        isMultiView,isSerialMultiMov,isExternal,isRemote,nowstr,logfile,...
        errfile,modelChainID,isContiguous,calibrationfileLcl] = ...
        myparse(varargin,...
        'ivw',[],...
        'targets',[],...
        'frm0',[],...
        'frm1',[],...
        'cropRoi',[],... % serialmode: either [], or [nmovset x 4]
        'tMFTConc',[],...
        'lblfileLcl','',...
        'movfileLcl',{},...
        'trxfileLcl',{},...
        'trkfileLcl',{},...
        'trkoutdirLcl',{},...
        'rootdirLcl','',...
        'isMultiView',false,... % flag of whether we want to track all views in one track job
        'isSerialMultiMov',false,... % see class comments
        'isExternal',[],...
        'isRemote',[],... % whether remote files are different from local files; if remote, remote filesys is assumed to be *nix
        'nowstr',[],...
        'logfile',{},...
        'errfile',{},...
        'modelChainID','',...
        'isContiguous',[],...
        'calibrationfileLcl',[] ...
        );

      obj.tObj = tObj;
      obj.nTotalView = obj.tObj.lObj.nview;
      if isempty(isMultiView),
        obj.tfmultiview = numel(ivw) > 1; 
        % see assert below
      else
        obj.tfmultiview = isMultiView;
      end
      if isempty(ivw),
        if obj.tfmultiview,
          obj.ivw = 1:obj.nTotalView;
        else
          obj.ivw = 1;
        end
      else
        obj.ivw = ivw;
      end
      if obj.tfmultiview
        % AL20200227: It appears we require this assert b/c if
        % tfmultiview==true we do not pass the -view arg to APT_interf => 
        % all views will be tracked.
        assert(isequal(obj.ivw(:)',1:obj.nTotalView));
      end
      if isempty(isContiguous),
        if isempty(tMFTConc),
          isContiguous = true;
          assert(~isempty(frm0) && ~isempty(frm1));
        elseif isempty(frm0) || isempty(frm1),
          isContiguous = false;
          assert(~isempty(tMFTConc));
        else
          isContiguous = TrackJob.isMFTContiguous(tMFTConc,frm0,frm1);
        end
      end
      obj.isContiguous = isContiguous;
      obj.calibrationfileLcl = calibrationfileLcl;
      obj.setBackEnd(backend);
      obj.nView = numel(obj.ivw);
      obj.tftrx = obj.tObj.lObj.hasTrx;
      obj.tf2stg = obj.tObj.getNumStages()>1;
      if isempty(nowstr),
        obj.nowstr = datestr(now,'yyyymmddTHHMMSS');
      else
        obj.nowstr = nowstr;
      end
      
      if isempty(modelChainID),
        obj.modelChainID = obj.tObj.trnName;
      else
        obj.modelChainID = modelChainID;
      end
      
      [trnstrs,modelFiles] = obj.tObj.getTrainStrModelFiles();
      if obj.tf2stg
        assert(numel(modelFiles)==2);
        obj.modelfile = modelFiles(:)';
        obj.trnstr = trnstrs(:)';
      else
        obj.modelfile = modelFiles(obj.ivw);
        obj.modelfile = obj.modelfile(:);
        obj.trnstr = trnstrs(obj.ivw);
        obj.trnstr = obj.trnstr(:);
      end
      
      % currently need to specify frm0 and frm1
      %assert(~isempty(frm0) && ~isempty(frm1));
      obj.frm0 = frm0;
      obj.frm1 = frm1;
            
      if isempty(isExternal),
        obj.tfexternal = ~isempty(movfileLcl);
      else
        obj.tfexternal = isExternal;
      end
            
      if isempty(isRemote),
        obj.tfremote = obj.backend.type == DLBackEnd.AWS;
      else
        obj.tfremote = isRemote;
      end
      
      if isSerialMultiMov
        assert(~obj.tfmultiview);
        assert(obj.tfexternal);
        assert(~obj.tfremote);
      end
      obj.tfserialmultimov = isSerialMultiMov;

      if isempty(rootdirLcl),
        obj.rootdirLcl = obj.tObj.lObj.DLCacheDir;
      else
        obj.rootdirLcl = rootdirLcl;
      end
      
      % obj.dmc*: shape important
      if obj.tf2stg
        obj.dmcRem = obj.tObj.trnLastDMC(:)';
        obj.dmcLcl = obj.dmcRem;
      else
        obj.dmcRem = obj.tObj.trnLastDMC(obj.ivw);
        obj.dmcRem = obj.dmcRem(:);
        obj.dmcLcl = obj.dmcRem;
      end      
      if obj.tfremote,
        for i = 1:numel(obj.dmcLcl);
          obj.dmcLcl(i) = obj.dmcRem(i).copy();
          obj.dmcLcl(i).rootDir = obj.rootdirLcl;
        end
      end
      obj.rootdirRem = obj.dmcRem(1).rootDir;
      
      if isempty(lblfileLcl),
        lblfileLcl = unique({obj.dmcLcl.lblStrippedLnx});
        assert(isscalar(lblfileLcl));
        obj.lblfileLcl = lblfileLcl{1};
      else
        obj.lblfileLcl = lblfileLcl;
      end
      obj.lblfileRem = obj.dmcRem.lblStrippedLnx;
      
      obj.trkoutdirRem = cell(1,obj.nView);
      for i = 1:obj.nView,
        obj.trkoutdirRem{i} = obj.dmcRem(i).dirTrkOutLnx;
      end
      
      %if obj.tfexternal
        % .trkoutdirLcl will be unused
        %obj.trkoutdirLcl = cell(1,obj.nView);
      %else
      
      % in case of obj.tfexternal, trkoutdirLcl still used for trk errfile
      nset = 1 + double(obj.tf2stg);
      if isempty(trkoutdirLcl)
        obj.trkoutdirLcl = arrayfun(@(x)x.dirTrkOutLnx,obj.dmcLcl,'uni',0);
      else
        obj.trkoutdirLcl = trkoutdirLcl;
      end
      szassert(obj.trkoutdirLcl,[obj.nView nset]);
      
      if obj.tfexternal,
        assert(~isempty(movfileLcl) && ~isempty(trkfileLcl));
        obj.movfileLcl = movfileLcl;
        obj.trkfileLcl = trkfileLcl;
        if obj.tftrx,
          %assert(~isempty(trxids));
          obj.trxids = trxids;
          obj.trxfileLcl = trxfileLcl;
          obj.trkfileLcl = obj.trkfileLcl(:);
        elseif obj.tf2stg
          obj.trxids = {};
          obj.trxfileLcl = {};
          obj.trkfileLcl = obj.trkfileLcl(:)'; % shape important
          if isscalar(obj.trkfileLcl)
            warningNoTrace('Two-stage tracker: using default stage1 trkfilename.');
            [tflP,tflF,tflE] = fileparts(obj.trkfileLcl);
            tflStg1 = fullfile(tflP,[tflF '_stg1' tflE]);
            obj.trkfileLcl = {tflStg1 obj.trkfileLcl{1}};
          end
        else
          obj.trxids = {};
          obj.trxfileLcl = {};
          obj.trkfileLcl = obj.trkfileLcl(:);
        end
        
        if obj.tfserialmultimov
          nserial = numel(obj.movfileLcl);
          obj.nSerialMov = nserial;
          assert(size(obj.trkfileLcl,1)==nserial);
          if obj.tftrx
            assert(numel(obj.trxfileLcl)==nserial);
          end
        else
          assert(size(obj.trkfileLcl,1)==obj.nView && ...
                 numel(obj.movfileLcl)==obj.nView);
          if obj.tftrx
            assert(numel(obj.trxfileLcl)==obj.nView);
          end
        end
      else
        assert(~isempty(tMFTConc));
        obj.settMFTConc(tMFTConc);
      end

      if isempty(cropRoi),
        obj.cropRoi = [];
      elseif obj.tfserialmultimov
        if ~isempty(cropRoi)
          szassert(cropRoi,[obj.nSerialMov 4]);
        end
        obj.cropRoi = cropRoi;
      else
        if iscell(cropRoi),
          assert(numel(cropRoi)==1);
          cropRoi = cropRoi{1};
        end
        szassert(cropRoi,[obj.nTotalView 4]);
        obj.cropRoi = cropRoi(obj.ivw,:);
      end
      
      obj.setLocalFiles();
      
      obj.setRemoteFiles();

      obj.setPartTrkFiles();
            
      obj.setId();
      
      obj.setLogErrFiles('logfile',logfile,'errfile',errfile);      
      
    end
    
    function prepareFiles(obj)
      
      obj.checkCreateDirs();
      obj.checkLocalFiles();
      if ~obj.isContiguous,
        obj.createLocalListFile();
      end
      obj.checkCreateRemotes();
      obj.deletePartTrkFiles();
      
    end
    
    function [codestr,containerName] = setCodeStr(obj,varargin)
      
      [bsubargs,sshargs,baseargs,singargs,...
        dockerargs,mntpaths,useLogFlag,...
        condaargs] = ...
        myparse(varargin,...
        'bsubargs',{},'sshargs',{},...
        'baseargs',{},'singargs',{},...
        'dockerargs',{},'mntpaths',{},'useLogFlag',ispc,...
        'condaargs',{});
      containerName = obj.id;
      baseargs = obj.getBaseArgs(baseargs);

      if obj.tf2stg
        fileargs = struct('trnID',obj.modelChainID,...
          'cache',obj.rootdirRem,...
          'dllbl',obj.lblfileRem,...
          'errfile',obj.errfile,...
          'nettype',obj.tObj.trnNetType,...
          'netmode',obj.tObj.trnNetMode,...
          'nettypeStage1',obj.tObj.stage1Tracker.trnNetType,...
          'netmodeStage1',obj.tObj.stage1Tracker.trnNetMode,...
          'movtrk',{obj.movfileRem},...
          'outtrk',{obj.trkfileRem});
      else
        fileargs = struct('trnID',obj.modelChainID,...
          'cache',obj.rootdirRem,...
          'dllbl',obj.lblfileRem,...
          'errfile',obj.errfile,...
          'nettype',obj.tObj.trnNetType,...
          'netmode',obj.tObj.trnNetMode,...
          'movtrk',{obj.movfileRem},...
          'outtrk',{obj.trkfileRem});
      end
      
      switch obj.backend.type,
        
        case DLBackEnd.Bsub,
          
          bsubargs = [bsubargs,{'outfile' obj.logfile}];
          % trnID,cache,dllbl,errfile,...
          % nettype,movtrk,outtrk,frm0,frm1,
          obj.codestr = DeepTracker.trackCodeGenSSHBsubSing(...
            fileargs,...
            obj.frm0,obj.frm1,...
            'baseargs',baseargs,'singArgs',singargs,'bsubargs',bsubargs,...
            'sshargs',sshargs);
          
        case DLBackEnd.Docker,
          
          if useLogFlag
            baseargs = [baseargs {'log_file' obj.logfile}]; 
          end
          obj.codestr = ...
            DeepTracker.trackCodeGenDocker(obj.backend,...
            fileargs,...
            obj.frm0,obj.frm1,...
            'baseargs',baseargs,'mntPaths',mntpaths,...
             'containerName',obj.containerName,...
             'dockerargs',dockerargs);
          
        case DLBackEnd.Conda,
                
          [obj.codestr] = ...
            DeepTracker.trackCodeGenConda(...
            fileargs,...
            obj.frm0,obj.frm1,...
            'baseargs',[baseargs,{'filesep',obj.tObj.filesep}],...
            'outfile',obj.logfile,...
            'condaargs',condaargs);
          
        case DLBackEnd.AWS,
          
          codestrRem = ...
            DeepTracker.trackCodeGenAWS(...
            fileargs,...
            obj.frm0,obj.frm1,...
            baseargs);
          
          logfilelnx = regexprep(obj.logfile,'\\','/'); % maybe this should be generated for lnx upstream
          obj.codestr = obj.backend.awsec2.sshCmdGeneralLogged(codestrRem,logfilelnx);
           
        otherwise
          error('not implemented back end %s',obj.backend.type);
          
      end
      codestr = obj.codestr;
    end
    
    function [logfiles,errfiles,trkfiles,partfiles,movfiles] = ...
        getMonitorArtifacts(objArr)
      % Generate file/artifact list for BgTrackMonitor.
      %
      % objArr: [nMovJobs x nViewJobs] array of TrackJobs
      % 
      % *files: with shapes as expected by BgTrackWOrkerObj.initFiles.
      % logfiles, errfiles: [nMovJobs x nViewJobs]
      % trkfiles, partfiles: [nMovs x nViews x nStages]
      % movfiles: [nMovs x nViews]      
      
      [nMovJobs,nViewJobs] = size(objArr);
      logfiles = reshape({objArr.logfile},size(objArr));
      errfiles = reshape({objArr.errfile},size(objArr));
      
      if objArr(1).tfmultiview % each job in objArr tracks across views
        assert(nViewJobs==1);
        movfiles = {objArr.movfileLcl};
        movfiles = cellfun(@(x)x(:)',movfiles,'uni',0);
        movfiles = cat(1,movfiles{:});
        
        trkfiles = {objArr.trkfileRem};
        trkfiles = cat(3,trkfiles{:}); % [nView x nset x nMovs]
        trkfiles = permute(trkfiles,[3 1 2]);
        partfiles = {objArr.parttrkfileRem};
        partfiles = cat(3,partfiles{:});
        partfiles = permute(partfiles,[3 1 2]);
        
      elseif objArr(1).tfserialmultimov % each job in objArr tracks across movs
        assert(nMovJobs==1);
        movfiles = {objArr.movfileLcl};
        movfiles = cellfun(@(x)x(:),movfiles,'uni',0);
        movfiles = cat(2,movfiles{:});
        
        trkfiles = {objArr.trkfileRem};
        trkfiles = cat(3,trkfiles{:}); % [nSerialMov x nset x nViews]
        trkfiles = permute(trkfiles,[1 3 2]);
        partfiles = {objArr.parttrkfileRem};
        partfiles = cat(3,partfiles{:});
        partfiles = permute(partfiles,[1 3 2]);
      else
        movfiles = cat(1,objArr.movfileLcl);
        movfiles = reshape(movfiles,size(objArr));

        %trkfiles = {objArr.trkfileRem}; % each el is a row [1 x nset]
        trkfiles = cat(1,objArr.trkfileRem); % [nMov*nView x nset]
        trkfiles = reshape(trkfiles,nMovJobs,nViewJobs,[]);
        partfiles = cat(1,objArr.parttrkfileRem); 
        partfiles = reshape(partfiles,nMovJobs,nViewJobs,[]); 
      end
    end
    
    function trkfiles = getTrkFilesSingleMov(objArr)
      % Get trkfiles produced by an array of TrackJobs for a single movie
      %
      % objArr: array of TrackJobs all for a single movie. All elements
      % must have same tfmultiview, tfserialmultimov, tf2stg etc.
      %
      % trkfiles: [nview] vector of trkfiles for movie produced by
      % TrackJobs. trkfiles should contain nview elements corresponding to
      % each view.
      
      obj1 = objArr(1);
      if obj1.tfserialmultimov
        assert(false,'Unsupported');
      elseif obj1.tfmultiview
        % objArr should prob be scalar here
        % Each el of objArr.trkfileLcl should be a column
        trkfiles = cat(1,objArr.trkfileLcl);
      elseif obj1.tf2stg
        trkfiles = cat(1,objArr.trkfileLcl);
        trkfiles = trkfiles(:,2); 
        % right now, just return final/stage1 trks
      else
        % typical nonscalar objArr case
        trkfiles = cat(1,objArr.trkfileLcl);        
      end
    end    
    
    function nframestrack = getNFramesTrack(obj,forcecompute)
      % nframestrack: [nmovsettrk] array. If tfmultiview, movs are assumed 
      % to have same number of frames across views and the view1 mov is 
      % used
      
      if nargin < 2,
        forcecompute = false;
      end
      if ~isempty(obj.nframestrack) && ~forcecompute,
        nframestrack = obj.nframestrack;
        return;
      end

      nmovset = obj.nmovsettrk;
      if ~obj.isContiguous,
        nframestrack = size(obj.tMFTConc,1);
        obj.nframestrack = nframestrack;
        return;
      end
      
      % get/compute frm0 (scalar) and frm1 ([nmovset])
      f0 = obj.frm0;
      f0(isnan(f0)) = 1;
      if isempty(f0),
        f0 = ones(nmovset,1);
      elseif numel(f0)==1,
        f0 = repmat(f0,nmovset,1);
      end
      f1 = obj.frm1;
      f1(isinf(obj.frm1)) = nan;
      if isempty(f1),
        f1 = nan(nmovset,1);
      elseif numel(f1) == 1,
        f1 = repmat(f1,nmovset,1);
      end
      lObj = obj.tObj.lObj;
      for imovset=1:nmovset
        % if .tfmultiview, this will go off the view1 mov
        if isnan(f1(imovset)),
          f1(imovset) = lObj.getNFramesMovFile(obj.movfileLcl{imovset});
        end
      end
      
      if ~obj.tftrx,
        nframestrack = f1-f0+1;
      else
        nframestrack = nan(nmovset,1);
        for imovset=1:nmovset
          % multiview-projs-with-trx currently not supported in APT so
          % .trxfileLcl is either a scalar cell or a [nmovset] cell
          [~,frm2trx] = obj.tObj.lObj.getTrx(obj.trxfileLcl{imovset});
          trxids1 = obj.getTrxIds(imovset);
          if isempty(trxids1),
            trxids1 = 1:size(frm2trx,2);
          end
          nframestrack(imovset) = sum(sum(frm2trx(f0(imovset):f1(imovset),trxids1)));
        end
      end
            
      obj.nframestrack = nframestrack;
      
    end
    
    function trxids = getTrxIds(obj,imovset)
      if iscell(obj.trxids),
        trxids = obj.trxids{imovset};
      else
        trxids = obj.trxids;
      end
    end
    
    function baseargsaug = getBaseArgs(obj,baseargsaug)
      if nargin < 2,
        baseargsaug  = {};
      end
      baseargsaug = [baseargsaug {'model_file' obj.modelfile}]; % this will be a cell hopefully that is ok! 
      if ~isempty(obj.cropRoi),
        baseargsaug = [baseargsaug {'croproi' obj.cropRoi}]; 
      end
      if ~obj.tfmultiview,
        baseargsaug = [baseargsaug {'view' obj.ivw}]; % 1-based OK
      end
      if obj.tftrx,
        baseargsaug = [baseargsaug {'trxtrk' obj.trxfileRem 'trxids' obj.trxids}]; 
      end
      if ~obj.isContiguous,
        baseargsaug = [baseargsaug {'listfile' obj.listfileRem}]; 
      end
    end
    
    function checkLocalFiles(obj)
      
      for i=1:numel(obj.movfileLcl)
        if ~exist(obj.movfileLcl{i},'file'),
          error('Movie file %s does not exist',obj.movfileLcl{i});
        end
      end
      if obj.tftrx,
        for i=1:numel(obj.trxfileLcl)
          if ~exist(obj.trxfileLcl{i},'file'),
            error('Trx file %s does not exist',obj.trxfileLcl{i});
          end
        end
      end
      
    end

    function setDefaultLocalListFile(obj)
      % I'm not sure when this has more than one element
      %assert(numel(obj.trnstr)==1);
      trnstr0 = obj.trnstr{1};
      obj.listfilestr = [ 'TrackList_' trnstr0 '_' obj.nowstr '.json'];
      obj.listfileLcl = fullfile(obj.rootdirLcl,obj.listfilestr);
    end
    
    function createLocalListFile(obj)
      
      % {
      %   "movieFiles": [
      %     "/path/to/mov1.avi",
      %     "/path/to/mov2.avi"
      %   ],
      %   "trxFiles": [
      %     "/path/to/trx1.mat",
      %     "/path/to/trx2.mat"
      %   ],
      %   "toTrack": [
      %     [1,1,1],
      %     [1,1,2],
      %     [1,1,3],
      %     [2,1,[1,2501]]
      %     ]
      % }
      
      if obj.tftrx,
        args = {'trxFiles',obj.trxfileRem};
      else
        args = {};
      end
      if ispc && obj.backend.type == DLBackEnd.Conda
        args(end+1:end+2) = {'isWinBackend' true};
      end
      DeepTracker.trackWriteListFile(...
        obj.movfileRem,obj.movfileLcl,obj.tMFTConc,obj.listfileLcl,args{:});
    end
    
    function setBackEnd(obj,backend)
      obj.backend = backend;
      switch obj.backend.type,
        case DLBackEnd.AWS,
          aws = obj.backend.awsec2;
          obj.uploadfun = @aws.scpUploadOrVerifyEnsureDir;
          obj.mkdirRemFun = @aws.ensureRemoteDir;
          obj.rmRemFun = @aws.rmRemoteFile;
          sysCmdArgs = {'dispcmd' true 'failbehavior' 'err'};
          obj.downloadfun = @(varargin) aws.scpDownloadOrVerify(varargin{:},'sysCmdArgs',sysCmdArgs);
        
      end
      
    end
        
    function setId(obj)
      [~,movS] = fileparts(obj.movfileRem{1});
      obj.id = [movS '_' obj.trnstr{1} '_' obj.nowstr];      
    end
    
    function settMFTConc(obj,tMFTConc)
      % assumes there is only one video being tracked
      obj.tMFTConc = tMFTConc;
      obj.tMFTConc.mov = obj.tMFTConc.mov(:,obj.ivw);
      obj.movfileLcl = obj.tMFTConc.mov(1,:);
      obj.trxids = unique(tMFTConc.iTgt);
      if obj.tObj.lObj.hasTrx,
        obj.trxfileLcl = obj.tMFTConc.trxFile(1,obj.ivw);
      else
        obj.trxfileLcl = {};
      end
      obj.setDefaultTrkFiles();
    end  
    
    function setDefaultTrkFiles(obj)
      % Only called from settMFTConc => only one movieset to track
      % sets .trkfileLcl, .trkfilestr
      
      movs = obj.movfileLcl;
      trnstrs = obj.trnstr;
      [nview,nset] = size(trnstrs);
      
      obj.trkfileLcl = cell([nview nset]);
      obj.trkfilestr = cell([nview nset]);
      for ivw=1:nview
        mov = movs{ivw};
        [~,movS] = fileparts(mov);
        for iset=1:nset
          trnstr0 = obj.trnstr{ivw,iset};
          obj.trkfilestr{ivw,iset} = [movS '_' trnstr0 '_' obj.nowstr '.trk'];
          obj.trkfileLcl{ivw,iset} = fullfile(obj.trkoutdirLcl{ivw,iset},...
                                              obj.trkfilestr{ivw,iset});
        end
      end
    end
    
    function setPartTrkFiles(obj)
      obj.parttrkfileLcl = cellfun(@(x) [x,'.part'],obj.trkfileLcl,'Uni',0);
      obj.parttrkfileRem = cellfun(@(x) [x,'.part'],obj.trkfileRem,'Uni',0);      
    end
    
    function setLogErrFiles(obj,varargin)
      % backend needs to be set before making this call
      %
      % AL: possible dup with DeepModelChainOnDisk
      
      [logfile,errfile] = myparse(varargin,'logfile','','errfile',''); %#ok<PROPLC>
      obj.logdir = obj.trkoutdirRem{1};
      isremote = obj.tfremote;
      if isempty(logfile), %#ok<PROPLC>
        if isremote
          obj.logfile = [obj.logdir '/' obj.id '.log'];
        else
          obj.logfile = fullfile(obj.logdir,[obj.id '.log']);
        end
      else
        obj.logfile = logfile; %#ok<PROPLC>
      end
      if isempty(errfile), %#ok<PROPLC>
        if isremote
          obj.errfile = [obj.logdir '/' obj.id '.err'];
        else
          obj.errfile = fullfile(obj.logdir,[obj.id '.err']);
        end
      else
        obj.errfile = errfile; %#ok<PROPLC>
      end
      if isremote
        obj.ssfile = [obj.logdir '/' obj.id '.aptsnapshot'];
      else
        obj.ssfile = fullfile(obj.logdir,[obj.id '.aptsnapshot']);
      end
    end
    
    function setLocalFiles(obj,varargin)
      obj.setDefaultLocalListFile();
    end
    
    function setRemoteFiles(obj,varargin)
      % backend needs to be set
      
%       obj.modelfileRem = cell(1,obj.nView);
%       for i = 1:obj.nView,
%         obj.modelfileRem{i} = obj.dmcRem(i).lblStrippedLnx;
%       end
      
      obj.movfileRem = obj.movfileLcl;
      obj.trxfileRem = obj.trxfileLcl;
      obj.trkfileRem = obj.trkfileLcl;
      obj.listfileRem = obj.listfileLcl;
      if obj.tfremote,
        for i = 1:obj.nView,
          mov = obj.movfileLcl{i};
          [~,~,movE] = fileparts(mov);
          movsha = DeepTracker.getSHA(mov);
          movRemoteRel = [obj.remoteDataDirRel '/' movsha movE];
          obj.movfileRem{i} = [obj.remoteRoot '/' movRemoteRel];
          
          if obj.tftrx,
            trx = obj.trxfileLcl{i};
            trxsha = DeepTracker.getSHA(trx);
            trxRemoteRel = [obj.remoteDataDirRel '/' trxsha];
            obj.trxfileRem{i} = [obj.remoteRoot '/' trxRemoteRel];
          end
          
          trnstr0 = obj.trnstr{i};
          trkRemoteRel = [movsha '_' trnstr0 '_' obj.nowstr '.trk'];
          obj.trkfileRem{i} = [obj.trkoutdirRem{i} '/' trkRemoteRel];
          
        end
        obj.listfileRem = [obj.dmcRem.dirModelChainLnx '/' obj.listfilestr];
      end
      
    end
    
    function checkCreateDirs(obj)

      %if ~obj.tfexternal
      % still used if tfexternal for errfile
      TrackJob.checkCreateDir(obj.trkoutdirLcl,'trk cache dir');
      %end
      if obj.tfremote,
        obj.mkdirRemFun(obj.remoteDataDirRel,'descstr','data');
        for i=1:numel(obj.trkoutdirRem)
          obj.mkdirRemFun(obj.trkoutdirRem{i},'descstr','data','relative',false);
        end
      end
    end
    
    function checkCreateRemotes(obj)
      
      if ~obj.tfremote,
        return;
      end
            
      % errors out if fails
      obj.checkUploadFiles({obj.lblfileLcl},{obj.lblfileRem},'Lbl file');
      obj.checkUploadFiles(obj.movfileLcl,obj.movfileRem,'Movie file');
      if obj.tftrx,
        obj.checkUploadFiles(obj.trxfileLcl,obj.trxfileRem,'Trx file');
      end
      if ~obj.isContiguous,
        obj.checkUploadFiles({obj.listfileLcl},{obj.listfileRem},'Track list json file');
      end
      
    end
    
    function checkUploadFiles(obj,fileLcl,fileRem,varargin)
      
      if ~obj.tfremote,
        return;
      end
      
      for i = 1:numel(fileLcl),
        obj.uploadfun(fileLcl{i},fileRem{i},varargin{:});
      end
      
    end
    
    function tfsucc = downloadRemoteResults(obj)
      
      tfsucc = true;
      if ~obj.tfremote,
        return;
      end
      
      tfsucc = obj.downloadFiles(obj.trkfileRem,obj.trkfileLcl);
      
    end
    
    function tfsucc = downloadFiles(obj,fileRem,fileLcl)
      tfsucc = true;
      for i = 1:numel(fileLcl),
        tfsucc = tfsucc && obj.downloadfun(fileRem{i},fileLcl{i});
      end      
    end
    
    function deletePartTrkFiles(obj)
      
      TrackJob.checkDeleteFiles(obj.parttrkfileLcl,'partial tracking result');
      if obj.tfremote,
        obj.rmRemFun(obj.parttrkfileRem,'partial tracking result');
      end

    end
    
    function v = get.nmovsettrk(obj)
      if obj.tfserialmultimov
        v = numel(obj.movfileLcl);
      else
        v = 1;
      end
    end

    function v = get.trkfile(obj)
      if obj.tfmultiview || obj.tfserialmultimov || obj.tf2stg
        v = obj.trkfileLcl;
      else
        v = obj.trkfileLcl{1};
      end      
    end
    
    function v = get.containerName(obj)
      
      v = obj.id;
    
    end
    
    function v = get.movfile(obj)
      if obj.tfmultiview || obj.tfserialmultimov
        v = obj.movfileLcl;
      else
        v = obj.movfileLcl{1};
      end
    end    
    
%     function v = get.trkfilelocal(obj)
%       if obj.tfmultiview || obj.tfserialmultimov
%         v = obj.trkfileLcl;
%       else
%         v = obj.trkfileLcl{1};
%       end
%     end
%     
%     function v = get.trkfileremote(obj)
%       if obj.tfmultiview || obj.tfserialmultimov
%         v = obj.trkfileRem;
%       else
%         v = obj.trkfileRem{1};
%       end
%     end
    
    
%     function v = get.parttrkfilelocal(obj)
%       if obj.tfmultiview || obj.tfserialmultimov
%         v = obj.parttrkfileLcl;
%       else
%         v = obj.parttrkfileLcl{1};
%       end
%     end
    
%     function v = get.parttrkfileremote(obj)
%       if obj.tfmultiview || obj.tfserialmultimov
%         v = obj.parttrkfileRem;
%       else
%         v = obj.parttrkfileRem{1};
%       end
%     end
    
%     function v = get.trxlocal(obj)
%       if obj.tfmultiview,
%         v = obj.trxfileLcl;
%       else
%         v = obj.trxfileLcl{1};
%       end
%     end
    
%     function v = get.trxremote(obj)
%       if obj.tfmultiview,
%         v = obj.trxfileRem;
%       else
%         v = obj.trxfileRem{1};
%       end
%     end
    
    function v = get.snapshotfile(obj)
      v = obj.ssfile;
    end
    
    function extradirs = getMountDirs(obj)
      
      extradirs = {};
      for i = 1:numel(obj.movfileRem),
        movdir = fileparts(obj.movfileRem{i});
        extradirs{end+1} = movdir; %#ok<AGROW>
      end
      for i = 1:numel(obj.trkfileRem),
        trkdir = fileparts(obj.trkfileRem{i});
        extradirs{end+1} = trkdir; %#ok<AGROW>
      end
      extradirs = unique(extradirs);
      
    end
    
  end
  
  methods (Static)
    
    function checkDeleteFiles(filelocs,desc)
      if nargin < 2 || ~ischar(desc),
        desc = 'file';
      end
      for i = 1:numel(filelocs),
        if exist(filelocs{i},'file'),
          fprintf('Deleting %s %s',desc,filelocs{i});
          delete(filelocs{i});
        end
        if exist(filelocs{i},'file'),
          error('Failed to delete %s: file still exists',filelocs{i});
        end
      end
    end
    
    function checkCreateDir(dirlocs,desc,varargin)
      
      if nargin < 2 || ~ischar(desc),
        desc = 'dir';
      end
      assert(isempty(varargin));
      for i = 1:numel(dirlocs),
        if exist(dirlocs{i},'dir')==0
          [succ,msg] = mkdir(dirlocs{i});
          if ~succ
            error('Failed to create %s %s: %s',desc,dirlocs{i},msg);
          else
            fprintf(1,'Created %s: %s\n',desc,dirlocs{i});
          end
        end
      end      
    end

    function isContiguous = isMFTContiguous(tMFTConc,frm0,frm1)
      isContiguous = true;
      % assume that each movie for view1 is unique
      [m,~,idxm] = unique(tMFTConc.mov(:,1));
      for mi = 1:numel(m),
        idx1 = find(idxm==mi);
        [t,~,idxt] = unique(tMFTConc.iTgt(idx1));
        for ti = 1:numel(t),
          idx2 = idxt==ti;
          idxcurr = idx1(idx2);
          if ~isempty(setxor(tMFTConc.frm(idxcurr),(frm0:frm1)')),
            isContiguous = false;
            break;
          end
        end
        if ~isContiguous,
          break;
        end
      end
    end

  end
  
end