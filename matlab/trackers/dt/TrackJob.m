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


  properties
    
    tObj = [];
    backend = [];

    trnstr = {}; % cellstr, first output of DeepTracker.getTrkFileTrnStr; corresponding to .ivw
    nowstr = 'UNIQUETAG';
    id = 'TAG_ANN';
    modelChainID = '';
    remoteRoot = '/home/ubuntu';
    remoteDataDirRel = 'data';

    modelfile = {}; % cellstr, [nView] corresponding to .ivw
    logdir = '';
    logfile = '';
    errfile = '';
    
    % AL: Ideally we would enforce these all to be [nmovsettrk x nView]
    lblfileLcl = '';
    movfileLcl = {}; % cellstr, [nView] corresponding to .ivw; or if tfserialmultimov, [nSerialMov]
    trxfileLcl = {}; % unused if .tftrx is false. If .tftrx is true:
                     % if ~tfserialmultimov, cellstr [nView] corresponding to .ivw (currently APT only supports single-view projs with trx);
                     % if tfserialmultimov, [nSerialMov] corresponding to .movfileLcl
    trkfileLcl = {}; % cellstr, [nView] corresponding to .ivw; or if tfserialmultimov, [nSerialMov]
    trkoutdirLcl = {}; % cellstr, [nView] corresponding to .ivw. UNUSED if .tfexternal
    parttrkfileLcl = {}; % cellstr, [nView] corresponding to .ivw; or if tfserialmultimov, [nSerialMov]
    trkfilestr = {}; % cellstr, [nView] shortnames for trkfileLcl. UNUSED if .tfexternal
    rootdirLcl = '';
    
    lblfileRem = '';
    movfileRem = {};
    trxfileRem = {};
    trkfileRem = {};
    trkoutdirRem = {};
    parttrkfileRem = {};
    rootdirRem = '';
        
    tfmultiview = false;
    tfserialmultimov = false;
    tfexternal = false;
    tftrx = false;
    tfremote = false;

    dmcLcl = []; % [nView] dmc
    dmcRem = [];
    
    tMFTConc = [];
    trxids = {};
    frm0 = []; % currently must be scalar
    frm1 = []; % currently must be scalar
    cropRoi = []; % either [], or [nView x 4], or (if tfserialmultimov) [nSerialMov x 4]
    nframestrack = []; % after set via getNFramesTrack, [nmovsettrk]
    
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
    movlocal
    movremote
    trkfilelocal
    trkfileremote
    parttrkfilelocal
    parttrkfileremote
    trxlocal
    trxremote
    snapshotfile
        
  end
  
  methods
    
    function obj = TrackJob(tObj,backend,varargin)
      
      [ivw,trxids,frm0,frm1,cropRoi,tMFTConc,...
        lblfileLcl,movfileLcl,trxfileLcl,trkfileLcl,trkoutdirLcl,rootdirLcl,...
        isMultiView,isSerialMultiMov,isExternal,isRemote,nowstr,logfile,...
        errfile,modelChainID] = ...
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
        'modelChainID',''...
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
      
      obj.setBackEnd(backend);
      obj.nView = numel(obj.ivw);
      obj.tftrx = obj.tObj.lObj.hasTrx;
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
      
      [trnstrs,modelFiles] = obj.tObj.getTrkFileTrnStr();
      obj.modelfile = modelFiles(obj.ivw);
      obj.trnstr = trnstrs(obj.ivw);
      
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
      
      obj.dmcRem = obj.tObj.trnLastDMC(obj.ivw);
      obj.dmcLcl = obj.dmcRem;
      if obj.tfremote,
        for i = 1:obj.nView,
          obj.dmcLcl(i) = obj.dmcRem(i).copy();
          obj.dmcLcl(i).rootDir = obj.rootdirLcl;
        end
      end
      obj.rootdirRem = obj.dmcRem.rootDir;
      
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
      
      if obj.tfexternal
        % .trkoutdirLcl will be unused
        obj.trkoutdirLcl = cell(1,obj.nView);
      else
        if isempty(trkoutdirLcl),
          obj.trkoutdirLcl = cell(1,obj.nView);
          for i = 1:obj.nView,
            obj.trkoutdirLcl{i} = obj.dmcLcl(i).dirTrkOutLnx;
          end
        else
          assert(numel(trkoutdirLcl)==obj.nView);
          obj.trkoutdirLcl = trkoutdirLcl;
        end
      end
      
      if obj.tfexternal,
        assert(~isempty(movfileLcl) && ~isempty(trkfileLcl));
        obj.movfileLcl = movfileLcl;
        obj.trkfileLcl = trkfileLcl;
        if obj.tftrx,
          %assert(~isempty(trxids));
          obj.trxids = trxids;
          obj.trxfileLcl = trxfileLcl;
        else
          obj.trxids = {};
          obj.trxfileLcl = {};
        end
        
        if obj.tfserialmultimov
          nserial = numel(obj.movfileLcl);
          obj.nSerialMov = nserial;
          assert(numel(obj.trkfileLcl)==nserial);
          if obj.tftrx
            assert(numel(obj.trxfileLcl)==nserial);
          end
        else
          assert(numel(obj.trkfileLcl)==obj.nView && ...
                 numel(obj.movfileLcl)==obj.nView);
          if obj.tftrx
            assert(numel(obj.trxfileLcl)==obj.nView);
          end
        end
      else
        % currently tracks from frm0 to frm1 for all targets and one video
        % this should be fixed!
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
      
      obj.setRemoteFiles();

      obj.setPartTrkFiles();
            
      obj.setId();
      
      obj.setLogErrFiles('logfile',logfile,'errfile',errfile);      
      
    end
    
    function prepareFiles(obj)
      
      obj.checkCreateDirs();
      obj.checkLocalFiles();
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

      switch obj.backend.type,
        
        case DLBackEnd.Bsub,
          
          bsubargs = [bsubargs,{'outfile' obj.logfile}];
          obj.codestr = DeepTracker.trackCodeGenSSHBsubSing(...
            obj.modelChainID,obj.rootdirRem,...
            obj.lblfileRem,...
            obj.errfile,obj.tObj.trnNetType,...
            obj.movfileRem,obj.trkfileRem,...
            obj.frm0,obj.frm1,...
            'baseargs',baseargs,'singArgs',singargs,'bsubargs',bsubargs,...
            'sshargs',sshargs);
          
        case DLBackEnd.Docker,
          
          if useLogFlag
            baseargs = [baseargs {'log_file' obj.logfile}]; 
          end
          obj.codestr = ...
            DeepTracker.trackCodeGenDocker(obj.backend,...
            obj.modelChainID,obj.rootdirRem,obj.lblfileRem,obj.errfile,obj.tObj.trnNetType,...
            obj.movfileRem,obj.trkfileRem,...
             obj.frm0,obj.frm1,...
             'baseargs',baseargs,'mntPaths',mntpaths,...
             'containerName',obj.containerName,...
             'dockerargs',dockerargs);
          
        case DLBackEnd.Conda,
                
          [obj.codestr] = ...
            DeepTracker.trackCodeGenConda(...
            obj.modelChainID,obj.rootdirRem,obj.lblfileRem,obj.errfile,obj.tObj.trnNetType,...
            obj.movfileRem,obj.trkfileRem,...
            obj.frm0,obj.frm1,...
            'baseargs',[baseargs,{'filesep',obj.tObj.filesep}],...
            'outfile',obj.logfile,...
            'condaargs',condaargs);
          
        case DLBackEnd.AWS,
          
          codestrRem = ...
            DeepTracker.trackCodeGenAWS(...
            obj.modelChainID,obj.rootdirRem,obj.lblfileRem,obj.errfile,obj.tObj.trnNetType,...
            obj.movfileRem,obj.trkfileRem,...
            obj.frm0,obj.frm1,...
            baseargs);
          
          logfilelnx = regexprep(obj.logfile,'\\','/'); % maybe this should be generated for lnx upstream
          obj.codestr = obj.backend.awsec2.sshCmdGeneralLogged(codestrRem,logfilelnx);
           
        otherwise
          error('not implemented back end %s',obj.backend.type);
          
      end
      codestr = obj.codestr;
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
      
      % get/compute frm0 (scalar) and frm1 ([nmovset])
      if isempty(obj.frm0) && isempty(obj.frm1),
        frm0 = 1;
        frm1 = nan(nmovset,1);
        lObj = obj.tObj.lObj;
        for imovset=1:nmovset
          % if .tfmultiview, this will go off the view1 mov
          frm1(imovset) = lObj.getNFramesMovFile(obj.movfileLcl{imovset});
        end
      else
        frm0 = obj.frm0;
        frm1 = repmat(obj.frm1,nmovset,1);
      end
      
      if ~obj.tftrx,
        nframestrack = frm1-frm0+1;
      else
        nframestrack = nan(nmovset,1);
        for imovset=1:nmovset
          % multiview-projs-with-trx currently not supported in APT so
          % .trxfileLcl is either a scalar cell or a [nmovset] cell
          [~,frm2trx] = obj.tObj.lObj.getTrx(obj.trxfileLcl{imovset});
          if isempty(obj.trxids),
            trxids = 1:size(frm2trx,2);
          else
            trxids = obj.trxids;
          end
          nframestrack(imovset) = sum(sum(frm2trx(frm0:frm1(imovset),trxids)));
        end
      end
            
      obj.nframestrack = nframestrack;
      
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
      % currently tracks from frm0 to frm1 for all targets and one video
      % this should be fixed!
      obj.tMFTConc = tMFTConc;
      obj.movfileLcl = obj.tMFTConc.mov(1,obj.ivw);
      obj.trxids = unique(tMFTConc.iTgt);
      if obj.tObj.lObj.hasTrx,
        obj.trxfileLcl = obj.tMFTConc.trxFile(1,obj.ivw);
      else
        obj.trxfileLcl = {};
      end
      obj.setDefaultTrkFiles();
    end  
    
    function setDefaultTrkFiles(obj)
      % Only called from settMFTConc
            
      obj.trkfilestr = cell(size(obj.movfileLcl));
      obj.trkfileLcl = cell(size(obj.movfileLcl));
      for i = 1:obj.nView,
        mov = obj.movfileLcl{i};
        trnstr0 = obj.trnstr{i};
        [~,movS] = fileparts(mov);
        obj.trkfilestr{i} = [movS '_' trnstr0 '_' obj.nowstr '.trk'];
        obj.trkfileLcl{i} = fullfile(obj.trkoutdirLcl{i},obj.trkfilestr{i});
      end
      
    end
    
    function setPartTrkFiles(obj)
      
      obj.parttrkfileLcl = cellfun(@(x) [x,'.part'],obj.trkfileLcl,'Uni',0);
      obj.parttrkfileRem = cellfun(@(x) [x,'.part'],obj.trkfileRem,'Uni',0);
      
    end

    
    function setLogErrFiles(obj,varargin)
      % backend needs to be set before making this call
      
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
    
    function setRemoteFiles(obj,varargin)
      % backend needs to be set
      
%       obj.modelfileRem = cell(1,obj.nView);
%       for i = 1:obj.nView,
%         obj.modelfileRem{i} = obj.dmcRem(i).lblStrippedLnx;
%       end
      
      obj.movfileRem = obj.movfileLcl;
      obj.trxfileRem = obj.trxfileLcl;
      obj.trkfileRem = obj.trkfileLcl;
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
      end
      
    end
    
    function checkCreateDirs(obj)

      if ~obj.tfexternal
        TrackJob.checkCreateDir(obj.trkoutdirLcl,'trk cache dir');
      end
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
      
      if obj.tfmultiview || obj.tfserialmultimov
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
    
    function v = get.trkfilelocal(obj)
      if obj.tfmultiview || obj.tfserialmultimov
        v = obj.trkfileLcl;
      else
        v = obj.trkfileLcl{1};
      end
    end
    
    function v = get.trkfileremote(obj)
      if obj.tfmultiview || obj.tfserialmultimov
        v = obj.trkfileRem;
      else
        v = obj.trkfileRem{1};
      end
    end
    
    
    function v = get.parttrkfilelocal(obj)
      if obj.tfmultiview || obj.tfserialmultimov
        v = obj.parttrkfileLcl;
      else
        v = obj.parttrkfileLcl{1};
      end
    end
    
    function v = get.parttrkfileremote(obj)
      if obj.tfmultiview || obj.tfserialmultimov
        v = obj.parttrkfileRem;
      else
        v = obj.parttrkfileRem{1};
      end
    end
    
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
    
    function checkCreateDir(dirlocs,desc)
      
      if nargin < 2 || ~ischar(desc),
        desc = 'dir';
      end
      
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

    
  end
  
end