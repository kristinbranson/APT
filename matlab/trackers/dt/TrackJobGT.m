classdef TrackJobGT < handle
  % TrackJob is good but it is specialized towards regular tracking; the gt
  % case represents mostly new code.
  
  properties
    backend
    dmcslcl % [nview]
    dmcsrem % [nview]
    isserial = true;
    
    nettype % scalar DLNetType 
    netmode % scalar DLNetMode
    codestr
    codestrlog % for docker
  end
  properties (Dependent) % artifacts for monitoring
    mntrLogfile
    mntrErrfile
    mntrOutfile
    mntrPrtfile
    trkOutdirLcl
    trkOutdirRem
  end
  methods 
    function v = get.mntrLogfile(obj)
      v = DeepModelChainOnDisk.getCheckSingle(obj.dmcsrem.trkLogLnx);
    end
    function v = get.mntrErrfile(obj)
      v = DeepModelChainOnDisk.getCheckSingle(obj.dmcsrem.trkErrfileLnx);
    end
    function v = get.mntrOutfile(obj)
      v = DeepModelChainOnDisk.getCheckSingle(obj.dmcsrem.gtOutfileLnx);
    end
    function v = get.mntrPrtfile(obj)
      v = DeepModelChainOnDisk.getCheckSingle(obj.dmcsrem.gtOutfilePartLnx);
    end
    function v = get.trkOutdirLcl(obj)
      v = DeepModelChainOnDisk.getCheckSingle(obj.dmcslcl.dirTrkOutLnx);
    end
    function v = get.trkOutdirRem(obj)
      v = DeepModelChainOnDisk.getCheckSingle(obj.dmcsrem.dirTrkOutLnx);
    end
  end
  
  methods
    function obj = TrackJobGT(be,dmclcl,dmcrem,net,netmde)
      assert(numel(dmclcl)==numel(dmcrem));
      obj.backend = be;
      obj.dmcslcl = dmclcl;
      obj.dmcsrem = dmcrem;
      obj.nettype = net;
      obj.netmode = netmde;
    end
    function checkCreateDirs(obj)
      % Similar to TrackJob

      TrackJob.checkCreateDir({obj.trkOutdirLcl},'trk cache dir');
      if obj.dmcsrem.isRemote
        be = obj.backend;
        assert(be.type==DLBackEnd.AWS);
        % Should prob be backend meth
        be.awsec2.ensureRemoteDir(obj.trkOutdirRem,...
          'descstr','trk cache dir','relative',false);
      end
    end
    function codegen(obj,varargin)
      % aka setCodeStr. As a side effect, can set job-required state on
      % backend
      
      switch obj.backend.type
        case DLBackEnd.Bsub
          obj.codestr = obj.codegenSSHBsubSing();
          obj.codestrlog = [];
        case DLBackEnd.Docker
          % gpuid used in codegen. Could do this earlier, but the number of
          % GPUs requested is dependent on TrackJobGT, eg whether the job
          % is serial-across-views or parallel.
          gpuids = obj.backend.getFreeGPUs(1); % always serially for now
          if isempty(gpuids)
            error('No GPUs with sufficient RAM available locally');
          end            
          [obj.codestr,obj.codestrlog] = obj.codegenDocker();
        case DLBackEnd.AWS
        case DLBackEnd.Conda
      end
    end
    function codebase = codegenBase(obj,baseargs)
      dmc1 = obj.dmcsrem;
      cache = dmc1.getRootDir;
      dlconfig = DeepModelChainOnDisk.getCheckSingle(dmc1.trainConfigLnx);
      errfile = DeepModelChainOnDisk.getCheckSingle(dmc1.trkErrfileLnx);
      gtoutfile = DeepModelChainOnDisk.getCheckSingle(dmc1.gtOutfileLnx);
      trnID = DeepModelChainOnDisk.getCheckSingle(dmc1.modelChainID);

      codebase = DeepTracker.trackCodeGenBaseGTClassify(trnID,cache,dlconfig,...
        gtoutfile,errfile,obj.nettype,baseargs{:});
    end
    function baseargs = codegenBaseArgs(obj)
      %assert(numel(obj.dmcslcl)==1,'TODO: mv');
      
      dmc1 = obj.dmcsrem;
      mdl = regexprep(DeepModelChainOnDisk.getCheckSingle(dmc1.trainCurrModelLnx),'\.index$','');
      baseargs = {...
        'deepnetroot' obj.backend.getAPTDeepnetRoot ...
        'model_file' mdl ...
        }; 
    end
    function codestr = codegenSSHBsubSing(obj)
      % no -view arg; gtcompute serially across views
            
      dmc1 = obj.dmcsrem;
      logfile = DeepModelChainOnDisk.getCheckSingle(dmc1.trkLogLnx);
      ssfile = DeepModelChainOnDisk.getCheckSingle(dmc1.trkSnapshotLnx);
      aptroot = obj.backend.getAPTRoot;
      
      singimg = pick_singularity_image(obj.backend, obj.netmode) ;
      
      baseargs = obj.codegenBaseArgs();
      bsubargs = {'outfile' logfile};
      %sshargs = {};
      bindpaths = {dmc1.getRootDir; [aptroot '/deepnet']};
      %singBind = obj.genContainerMountPath('aptroot',aptroot);
      singargs = {'bindpath',bindpaths, 'singimg', singimg};
      repoSSscriptLnx = [aptroot '/matlab/repo_snapshot.sh'];
      repoSScmd = sprintf('"%s" "%s" > "%s"',repoSSscriptLnx,aptroot,ssfile);
      prefix = [DLBackEndClass.jrcprefix '; ' repoSScmd];
      sshargs = {'prefix' prefix};
        
      codebase = obj.codegenBase(baseargs);
      codesing = DeepTracker.codeGenSingGeneral(codebase,singargs{:});
      codebsub = DeepTracker.codeGenBsubGeneral(codesing,bsubargs{:});
      codestr = DeepTracker.codeGenSSHGeneral(codebsub,sshargs{:});      
    end
    function [codestr,logcmd] = codegenDocker(obj,varargin)
%       [gpuid] = myparse(varargin,...
%         'gpuid',[] ... % {} ...  %  'containerName','' ... 'useLogFlag', ispc ...
%         );
      
      %baseargs = [{'cache' cache} baseargs];
      %filequote = bed.getFileQuoteDockerCodeGen;
      
      dmc1 = obj.dmcsrem;
      logfile = DeepModelChainOnDisk.getCheckSingle(dmc1.trkLogLnx);
      be = obj.backend;
      aptroot = be.getAPTRoot;
      baseargs = obj.codegenBaseArgs();
      baseargs = [baseargs {'filequote' '"'}];
      bindpaths = {dmc1.getRootDir(); [aptroot '/deepnet']};

      gpuids = be.gpuids;
%       if useLogFlag
%         baseargs = [baseargs {'log_file' obj.logfile}];
%       end
      codebase = obj.codegenBase(baseargs);   
      containerName = sprintf('gt_%s',dmc1.getTrkTSstr());
      codestr = be.codeGenDockerGeneral(codebase,containerName,...
        'bindpath',bindpaths,'gpuid',gpuids);      
      logcmd = sprintf('%s logs -f %s &> "%s" &',...
                  be.dockercmd,containerName,logfile); 
      be.dockercontainername = containerName;
    end
    
  end
end