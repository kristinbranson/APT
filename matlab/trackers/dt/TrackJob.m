classdef TrackJob < handle

  properties
    
    tObj = [];
    backend = [];

    trnstr = {};
    nowstr = 'UNIQUETAG';
    id = 'TAG_ANN';
    modelChainID = '';
    remoteRoot = '/home/ubuntu';
    remoteDataDirRel = 'data';

    modelfile = {};
    logdir = '';
    logfile = '';
    errfile = '';
    
    lblfileLcl = '';
    movfileLcl = {};
    trxfileLcl = {};
    trkfileLcl = {};
    trkoutdirLcl = {};
    parttrkfileLcl = {};
    rootdirLcl = '';

    lblfileRem = '';
    movfileRem = {};
    trxfileRem = {};
    trkfileRem = {};
    trkoutdirRem = {};
    parttrkfileRem = {};
    rootdirRem = '';
    
    trkfilestr = {};
    
    tfmultiview = false;
    tfexternal = false;
    tftrx = false;
    tfremote = false;

    dmcLcl = [];
    dmcRem = [];
    
    tMFTConc = [];
    trxids = {};
    frm0 = [];
    frm1 = [];
    cropRoi = [];
    nframestrack = [];
    
    nView = 0; % number of views in this job
    ivw = [];
    nTotalView = 0; % total number of views in project
    
    uploadfun = @(varargin) [];
    mkdirRemFun = @(varargin) [];
    rmRemFun = @(varargin) [];
    
    logcmd = '';
    codestr = '';
    ssfile = '';

  end
  
  properties (Dependent)

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
        isMultiView,isExternal,isRemote,nowstr,logfile,errfile,modelChainID] = ...
        myparse(varargin,...
        'ivw',[],...
        'targets',[],...
        'frm0',[],...
        'frm1',[],...
        'cropRoi',[],...
        'tMFTConc',[],...
        'lblfileLcl','',...
        'movfileLcl',{},...
        'trxfileLcl',{},...
        'trkfileLcl',{},...
        'trkoutdirLcl',{},...
        'rootdirLcl','',...
        'isMultiView',false,... % flag of whether we want to track all views in one track job
        'isExternal',[],...
        'isRemote',[],... % whether remote files are different from local files
        'nowstr',[],...
        'logfile',{},...
        'errfile',{},...
        'modelChainID',''...
        );

      obj.tObj = tObj;
      obj.nTotalView = obj.tObj.lObj.nview;
      if isempty(isMultiView),
        obj.tfmultiview = numel(ivw) > 1;
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
      assert(~isempty(frm0) && ~isempty(frm1));
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
      
      if isempty(trkoutdirLcl),
        obj.trkoutdirLcl = cell(1,obj.nView);
        for i = 1:obj.nView,
          obj.trkoutdirLcl{i} = obj.dmcLcl(i).dirTrkOutLnx;
        end
      else
        assert(numel(trkoutdirLcl)==obj.nView);
        obj.trkoutdirLcl = trkoutdirLcl;
      end
      
      if obj.tfexternal,
        assert(~isempty(movfileLcl) && ~isempty(trkfileLcl));
        obj.movfileLcl = movfileLcl;
        obj.trkfileLcl = trkfileLcl;
        if obj.tftrx,
          assert(~isempty(trxids));
          obj.trxids = trxids;
          obj.trxfileLcl = trxfileLcl;
        else
          obj.trxids = {};
          obj.trxfileLcl = {};
        end
        assert(numel(obj.trkfileLcl)==obj.nView && ...
          numel(obj.movfileLcl)==obj.nView);
      else
        % currently tracks from frm0 to frm1 for all targets and one video
        % this should be fixed!
        assert(~isempty(tMFTConc));
        obj.settMFTConc(tMFTConc);
      end

      if isempty(cropRoi),
        obj.cropRoi = [];
      else
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
            obj.modelChainID,obj.rootDirRem,obj.lblfileRem,obj.errfile,obj.tObj.trnNetType,...
            obj.movfileRem,obj.trkfileRem,...
            obj.frm0,obj.frm1,...
            'baseargs',[baseargs,{'filesep',obj.tObj.filesep}],...
            'outfile',obj.logfile,...
            'condaargs',condaargs);
           
      end
      codestr = obj.codestr;
    end
    
    function nframestrack = getNFramesTrack(obj,forcecompute)

      if nargin < 2,
        forcecompute = false;
      end
      if ~isempty(obj.nframestrack) && ~forcecompute,
        nframestrack = obj.nframestrack;
        return;
      end
      
      if ~obj.tftrx,
        nframestrack = obj.frm1-obj.frm0+1;
      else
        [~,frm2trx] = obj.tObj.lObj.getTrx(obj.trxfileLcl{1});
        nframestrack = sum(sum(frm2trx(obj.frm0:obj.frm1,obj.trxids)));
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
      
      for i = 1:obj.nView,
        if ~exist(obj.movfileLcl{i},'file'),
          error('Movie file %s does not exist',obj.movfileLcl{i});
        end
        if obj.tftrx,
          if ~exist(obj.trxfileLcl{i},'file'),
            error('Trx file %s does not exist',obj.trxfileLcl{i});
          end
        end
      end
      
    end

    
    function setBackEnd(obj,backend)
      obj.backend = backend;
      switch obj.backend,
        case DLBackEnd.AWS,
          aws = obj.backend.awsec2;
          obj.uploadfun = @aws.scpUploadOrVerifyEnsureDir;
          obj.mkdirRemFun = @aws.ensureRemoteDir;
          obj.rmRemFun = @aws.rmRemoteFile;
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
      [logfile,errfile] = myparse(varargin,'logfile','','errfile',''); %#ok<PROPLC>
      obj.logdir = obj.trkoutdirRem{1};
      if isempty(logfile), %#ok<PROPLC>
        obj.logfile = fullfile(obj.logdir,[obj.id '.log']);
      else
        obj.logfile = logfile; %#ok<PROPLC>
      end
      if isempty(errfile), %#ok<PROPLC>
        obj.errfile = fullfile(obj.logdir,[obj.id '.err']);
      else
        obj.errfile = errfile; %#ok<PROPLC>
      end
      obj.ssfile = fullfile(obj.logdir,[obj.id '.aptsnapshot']);
    end
    
    function setRemoteFiles(obj,varargin)
      
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
          
          [~,trkfilestr,ext] = fileparts(obj.trkfileLcl{i}); %#ok<PROPLC>
          obj.trkfileRem{i} = fullfile(obj.trkoutdirRem{i},[trkfilestr,ext]); %#ok<PROPLC>
          
        end
      end
      
    end
    
    function checkCreateDirs(obj)

      TrackJob.checkCreateDir(obj.trkoutdirLcl,'trk cache dir');
      if obj.tfremote,
        obj.mkdirRemFun(obj.remoteDataDirRel,'descstr','data');
      end
      
      
    end
    
    function checkCreateRemotes(obj)
      
      if ~obj.tfremote,
        return;
      end
            
      % errors out if fails
      obj.checkUploadFiles(obj.lblfileLcl,obj.lblfileRem,'training file');
      obj.checkUploadFiles(obj.movfileLcl,obj.movfileRem,'trxfile');
      if obj.tftrx,
        obj.checkUploadFiles(obj.trxfileLcl,obj.trxfileRem,'trxfile');
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
    
    function deletePartTrkFiles(obj)
      
      TrackJob.checkDeleteFiles(obj.parttrkfileLcl,'partial tracking result');
      if obj.tfremote,
        obj.rmRemFun(obj.parttrkfileRem,'partial tracking result');
      end

    end
    

    function v = get.trkfile(obj)
      
      if obj.tfmultiview,
        v = obj.trkfileLcl;
      else
        v = obj.trkfileLcl{1};
      end
      
    end
    
    function v = get.containerName(obj)
      
      v = obj.id;
    
    end
    
    function v = get.movfile(obj)
      if obj.tfmultiview,
        v = obj.movfileLcl;
      else
        v = obj.movfileLcl{1};
      end
    end

    function v = get.movlocal(obj)
      if obj.tfmultiview,
        v = obj.movfileLcl;
      else
        v = obj.movfileLcl{1};
      end
    end
    
     function v = get.movremote(obj)
       if obj.tfmultiview,
         v = obj.movfileRem;
       else
         v = obj.movfileRem{1};
       end
     end
     
    function v = get.trkfilelocal(obj)
      if obj.tfmultiview,
        v = obj.trkfileLcl;
      else
        v = obj.trkfileLcl{1};
      end
    end
    
    function v = get.trkfileremote(obj)
      if obj.tfmultiview,
        v = obj.trkfileRem;
      else
        v = obj.trkfileRem{1};
      end
    end
    
    
    function v = get.parttrkfilelocal(obj)
      if obj.tfmultiview,
        v = obj.parttrkfileLcl;
      else
        v = obj.parttrkfileLcl{1};
      end
    end
    
    function v = get.parttrkfileremote(obj)
      if obj.tfmultiview,
        v = obj.parttrkfileRem;
      else
        v = obj.parttrkfileRem{1};
      end
    end
    
    function v = get.trxlocal(obj)
      if obj.tfmultiview,
        v = obj.trxfileLcl;
      else
        v = obj.trxfileLcl{1};
      end
    end
    
    function v = get.trxremote(obj)
      if obj.tfmultiview,
        v = obj.trxfileRem;
      else
        v = obj.trxfileRem{1};
      end
    end
    
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