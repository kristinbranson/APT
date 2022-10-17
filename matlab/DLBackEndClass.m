classdef DLBackEndClass < matlab.mixin.Copyable
  % 
  % APT Backends specify a physical machine/server where GPU code is run.
  %
  % * DLNetType: what DL net is run
  % * DLBackEndClass: where/how DL was run
  %
  % TODO: Design is solidifying. This should be a base class with
  % subclasses for backend types. The .type prop would be redundant against
  % the concrete type. Codegen methods should be moved out of DeepTracker
  % and into backend subclasses and use instance state (eg, docker codegen 
  % for current tag; bsub for specified .sif; etc). Conceptually this class 
  % would just be "DLBackEnd" and the enum/type would go away.
  
  properties (Constant)
    minFreeMem = 9000; % in MiB
    currentDockerImgTag = 'tf23_mmdetection';
    currentDockerImgRoot = 'bransonlabapt/apt_docker';
    
    RemoteAWSCacheDir = '/home/ubuntu/cacheDL';

    jrchost = 'login1.int.janelia.org';
    jrcprefix = ':'; % 'source /etc/profile';
    jrcprodrepo = '/groups/branson/bransonlab/apt/repo/prod';
  end
  properties
    type  % scalar DLBackEnd
    
    % scalar logical. if true, bsub backend runs code in APT.Root/deepnet. 
    % This path must be visible in the backend or else.
    %
    % Conceptually this could be an arbitrary loc.
    %
    % Applies only to bsub. Name should be eg 'bsubdeepnetrunlocal'
    deepnetrunlocal = true; 
    bsubaptroot = []; % root of APT repo for bsub backend running     
    jrcsimplebindpaths = 1; 
        
    awsec2 % used only for type==AWS
    awsgitbranch
    
    dockerapiver = '1.40'; % docker codegen will occur against this docker api ver
    dockerimgroot = DLBackEndClass.currentDockerImgRoot;
    % We have an instance prop for this to support running on older/custom
    % docker images.
    dockerimgtag = DLBackEndClass.currentDockerImgTag;
    dockerremotehost = '';
    gpuids = []; % for now used by docker/conda
    dockercontainername = []; % transient
    %dockershmsize = 512; % in m; passed in --shm-size
    
    condaEnv = 'APT'; % used only for Conda
  end
  properties (Dependent)
    filesep
    dockerimgfull % full docker img spec (with tag if specified)
  end
  
  methods % Prop access
    function v = get.filesep(obj)
      if obj.type == DLBackEnd.Conda,
        v = filesep;
      else
        v = '/';
      end
    end
    function v = get.dockerimgfull(obj)
      v = obj.dockerimgroot;
      if ~isempty(obj.dockerimgtag)
        v = [v ':' obj.dockerimgtag];
      end
    end
  end
 
  methods
    
    function obj = DLBackEndClass(ty,oldbe)
      if nargin > 1,
        % save state
        obj.deepnetrunlocal = oldbe.deepnetrunlocal;
        obj.awsec2 = oldbe.awsec2;
        obj.dockerimgroot = oldbe.dockerimgroot;
        obj.dockerimgtag = oldbe.dockerimgtag;
        obj.condaEnv = oldbe.condaEnv;
      end
      
      obj.type = ty;
    end
    
    function cmd = wrapBaseCommand(obj,basecmd,varargin)

      switch obj.type,
        case DLBackEnd.Bsub,
          cmd = obj.wrapBaseCommandBsub(basecmd,varargin{:});
        case DLBackEnd.Docker
          cmd = obj.codeGenDockerGeneral(basecmd,varargin{:});
        case DLBackEnd.Conda
          cmd = obj.wrapCommandConda(basecmd);
        case DLBackEnd.AWS
          cmd = obj.wrapCommandAWS(basecmd);
        otherwise
          error('Not implemented: %s',obj.type);
      end
    end

    function cmd = logCommand(obj,containerName,logfile)
      assert(obj.type == DLBackEnd.Docker);
      cmd = sprintf('%s logs -f %s &> "%s" &',...
        obj.dockercmd,containerName,logfile);
    end

    function v = ignore_local(obj)
      % this is useful for singularity, not needed for Docker, probably bad
      % for Conda
      if obj.type == DLBackEnd.Bsub,
        v = true;
      else
        v = false;
      end
    end

    function jobID = parseJobID(obj,res)
      switch obj.type
        case DLBackEnd.Bsub,
          jobID = DLBackEndClass.parseJobIDBsub(res);
        case DLBackEnd.Docker,
          jobID = DLBackEndClass.parseJobIDDocker(res);
        otherwise
          error('Not implemented: %s',obj.type);
      end
    end

    function [tfSucc,jobID] = run(obj,syscmds,varargin)

      [logcmds,cmdfiles,jobdesc] = myparse(varargin,'logcmds',{},'cmdfiles',{},'jobdesc','job');

      if ~isempty(cmdfiles),
        DLBackEndClass.writeCmdToFile(syscmds,cmdfiles,jobdesc);
      end
      njobs = numel(syscmds);
      tfSucc = false(1,njobs);
      tfSuccLog = true(1,njobs);
      jobID = cell(1,njobs);
      for ijob=1:njobs,
        fprintf(1,'%s\n',syscmds{ijob});
        if obj.type == DLBackEnd.Conda,
          [jobID{ijob},st,res] = parfevalsystem(syscmds{ijob});
          tfSucc(ijob) = st == 0;
        else
          [st,res] = system(syscmds{ijob});
          tfSucc(ijob) = st == 0;
          if tfSucc(ijob),
            jobID{ijob} = obj.parseJobID(res);
          end
        end
        if ~tfSucc(ijob),
          warning('Failed to spawn %s %d: %s',jobdesc,ijob,res{ijob});
        else
          jobidstr = jobID{ijob};
          if isnumeric(jobidstr),
            jobidstr = num2str(jobidstr);
          end
          fprintf('%s %d spawned, ID = %s\n\n',jobdesc,ijob,jobidstr);
        end
        if numel(logcmds) >= ijob,
          fprintf(1,'%s\n',logcmds{ijob});
          [st2,res2] = system(logcmds{ijob});
          tfSuccLog(ijob) = st2 == 0;
          if ~tfSuccLog(ijob),
            warning('Failed to spawn logging for %s %d: %s.',jobdesc,ijob,res2);
          end
        end
      end

    end

    function delete(obj)
      % AL 20191218
      % DLBackEndClass can now be deep-copied (see copyAndDetach below) as 
      % sometimes this is necessary for serialization eg to disk.
      % Since the mapping from obj<->resource is no longer 1-to-1, 
      % destructors should no longer shut down resources.
      %
      % See new shutdown() call.
      
      % pass
    end
    
    function shutdown(obj)
      if obj.type==DLBackEnd.AWS
        aws = obj.awsec2;
        if ~isempty(aws)
          fprintf(1,'Stopping AWS EC2 instance %s.',aws.instanceID);
          tfsucc = aws.stopInstance();
          if ~tfsucc
            warningNoTrace('Failed to stop AWS EC2 instance %s.',aws.instanceID);
          end
        end
      end
    end
    
    function obj2 = copyAndDetach(obj)
      % See notes in BGClient, BGWorkerObjAWS.
      %
      % Sometimes we want a deep-copy of obj that is sanitized for
      % eg serialization. This copy may still be largely functional (in the
      % case of BGWorkerObjAWS) or perhaps it can be 'reconstituted' at
      % load-time as here.
      
      assert(isscalar(obj));
      obj2 = copy(obj);
      if ~isempty(obj2.awsec2)
        obj2.awsec2.clearStatusFuns();
      end
    end
    
  end
  methods (Access=protected)
    
    function obj2 = copyElement(obj)
      % overload so that .awsec2 is deep-copied
      obj2 = copyElement@matlab.mixin.Copyable(obj);
      if ~isempty(obj.awsec2)
        obj2.awsec2 = copy(obj.awsec2);
      end
    end
    
  end
  
  methods
    
    function modernize(obj)
      if obj.type==DLBackEnd.Docker || obj.type==DLBackEnd.Bsub
        currentTag = DLBackEndClass.currentDockerImgTag;
        if ~strcmp(obj.dockerimgtag,currentTag)
          warningNoTrace('Updating backend to latest APT Docker image tag: %s\n',...
            currentTag);
          obj.dockerimgtag = currentTag;
        end
      end
      % 20211101 turn on by default
      obj.jrcsimplebindpaths = 1;
    end
    
    function testConfigUI(obj,cacheDir)
      % Test whether backend is ready to do; display results in msgbox
      
      switch obj.type,
        case DLBackEnd.Bsub,
          DLBackEndClass.testBsubConfig(cacheDir);
        case DLBackEnd.Docker
          obj.testDockerConfig();
        case DLBackEnd.AWS
          obj.testAWSConfig();
        case DLBackEnd.Conda
          obj.testCondaConfig();
        otherwise
          msgbox(sprintf('Tests for %s have not been implemented',obj.type),...
            'Not implemented','modal');
      end
    end
    
    function [tf,reason] = getReadyTrainTrack(obj)
      tf = false;
      if obj.type==DLBackEnd.AWS
        aws = obj.awsec2;
        
        didLaunch = false;
        if ~obj.awsec2.isConfigured || ~obj.awsec2.isSpecified,
          [tfsucc,instanceID,instanceType,reason,didLaunch] = ...
            obj.awsec2.selectInstance(...
            'canconfigure',1,'canlaunch',1,'forceselect',0);
          if ~tfsucc || isempty(instanceID),
            reason = sprintf('Problem configuring: %s',reason);
            return;
          end
        end

        
        [tfexist,tfrunning] = obj.awsec2.inspectInstance;
        if ~tfexist,
          uiwait(warndlg(sprintf('AWS EC2 instance %s could not be found or is terminated. Please configure AWS back end with a different AWS EC2 instance.',obj.awsec2.instanceID),'AWS EC2 instance not found'));
          reason = 'Instance could not be found.';
          obj.awsec2.ResetInstanceID();
          return;
        end
        
        tf = tfrunning;
        if ~tf
          if didLaunch,
            btn = 'Yes';
          else
            qstr = sprintf('AWS EC2 instance %s is not running. Start it?',obj.awsec2.instanceID);
            tstr = 'Start AWS EC2 instance';
            btn = questdlg(qstr,tstr,'Yes','Cancel','Cancel');
            if isempty(btn)
              btn = 'Cancel';
            end
          end
          switch btn
            case 'Yes'
              tf = obj.awsec2.startInstance();
              if ~tf
                reason = sprintf('Could not start AWS EC2 instance %s.',obj.awsec2.instanceID);
                return;
              end
            otherwise
              reason = sprintf('AWS EC2 instance %s is not running.',obj.awsec2.instanceID);
              return;
          end
        end
        
        [tfsucc] = obj.awsec2.waitForInstanceStart();
        if ~tfsucc,
          reason = 'Timed out waiting for AWS EC2 instance to be spooled up.';
          return;
        end
        
        reason = '';
      else
        tf = true;
        reason = '';
      end
    end
    
    function s = prettyName(obj)
      switch obj.type,
        case DLBackEnd.Bsub,
          s = 'JRC Cluster';
        case DLBackEnd.Docker,
          s = 'Local';
        otherwise
          s = char(obj.type);
      end
    end

    function v = isLocal(obj)
      v = isequal(obj.type,DLBackEnd.Docker) || isequal(obj.type,DLBackEnd.Conda);
    end
    
    function [gpuid,freemem,gpuInfo] = getFreeGPUs(obj,nrequest,varargin)
      % Get free gpus subject to minFreeMem constraint (see optional PVs)
      %
      % This sets .gpuids
      % 
      % gpuid: [ngpu] where ngpu<=nrequest, depending on if enough GPUs are available
      % freemem: [ngpu] etc
      % gpuInfo: scalar struct

      [dockerimg,minFreeMem,condaEnv,verbose] = myparse(varargin,...
        'dockerimg',obj.dockerimgfull,...
        'minfreemem',obj.minFreeMem,...
        'condaEnv',obj.condaEnv,...
        'verbose',0 ...
      ); %#ok<PROPLC>
      
      gpuid = [];
      freemem = 0;
      gpuInfo = [];
      aptdeepnet = APT.getpathdl;
      
      switch obj.type,
        case DLBackEnd.Docker
          basecmd = 'echo START; python parse_nvidia_smi.py; echo END';
          bindpath = {aptdeepnet}; % don't use guarded
          codestr = obj.codeGenDockerGeneral(basecmd,...
            'containername','aptTestContainer',...
            'bindpath',bindpath,...
            'detach',false);
          if verbose
            fprintf(1,'%s\n',codestr);
          end
          [st,res] = system(codestr);
          if st ~= 0,
            warning('Error getting GPU info: %s\n%s',res,codestr);
            return;
          end
        case DLBackEnd.Conda
          basecmd = sprintf('echo START && python %s%sparse_nvidia_smi.py && echo END',...
            aptdeepnet,obj.filesep);
          codestr = sprintf('activate %s && %s',condaEnv,basecmd);
          [st,res] = system(codestr);
          if st ~= 0,
            warning('Error getting GPU info: %s',res);
            return;
          end
        otherwise
          error('Not implemented');
      end
      
      res0 = res;
      res = regexp(res,'\n','split');
      res = strip(res);
      i0 = find(strcmp(res,'START'),1);
      if isempty(i0),
        warning('Could not find START of GPU info');
        disp(res0);
        return;
      end
      i0 = i0+1;
      i1 = find(strcmp(res(i0+1:end),'END'),1)+i0;
      res = res(i0+1:i1-1);
      ngpus = numel(res);      
      gpuInfo = struct;
      gpuInfo.id = zeros(1,ngpus);
      gpuInfo.totalmem = zeros(1,ngpus);
      gpuInfo.freemem = zeros(1,ngpus);
      for i = 1:ngpus,
        v = str2double(strsplit(res{i},','));
        gpuInfo.id(i) = v(1);
        gpuInfo.freemem(i) = v(2);
        gpuInfo.totalmem(i) = v(3);
      end      
      
      [freemem,order] = sort(gpuInfo.freemem,'descend');
      gpuid = gpuInfo.id(order);
      ngpu = find(freemem>=minFreeMem,1,'last'); %#ok<PROPLC>

      global FORCEONEJOB;
      if isequal(FORCEONEJOB,true),
        warning('Forcing one GPU job');
        ngpu = 1;
      end
      
      freemem = freemem(1:ngpu);
      gpuid = gpuid(1:ngpu);
      
      ngpureturn = min(ngpu,nrequest);
      gpuid = gpuid(1:ngpureturn);

      freemem = freemem(1:ngpureturn);
      
      obj.gpuids = gpuid;
    end
    
    function pretrack(obj,cacheDir,dmc,setStatusFcn)
      switch obj.type        
        case DLBackEnd.AWS
          obj.awsPretrack(dmc,setStatusFcn);
        case DLBackEnd.Bsub
          obj.bsubPretrack(cacheDir);
      end      
    end
    
    function r = getAPTRoot(obj)
      switch obj.type
        case DLBackEnd.Bsub
          r = obj.bsubaptroot;
        case DLBackEnd.AWS
          r = '/home/ubuntu/APT';
        case DLBackEnd.Docker
          r = APT.Root;          
        case DLBackEnd.Conda
          r = APT.Root;          
      end
    end
    function r = getAPTDeepnetRoot(obj)
      r = [obj.getAPTRoot '/deepnet'];
    end
    
  end
  
  methods (Static)

    function tfSucc = writeCmdToFile(syscmds,cmdfiles,jobdesc)

      if nargin < 3,
        jobdesc = 'job';
      end
      if ischar(syscmds),
        syscmds = {syscmds};
      end
      if ischar(cmdfiles),
        cmdfiles = {cmdfiles};
      end
      tfSucc = false(1,numel(syscmds));
      assert(numel(cmdfiles) == numel(syscmds));
      for i = 1:numel(syscmds),
        [fh,msg] = fopen(cmdfiles{i},'w');
        if isequal(fh,-1)
          warningNoTrace('Could not open command file ''%s'': %s',cmdfile,msg);
        else
          fprintf(fh,'%s\n',syscmds{i});
          fclose(fh);
          fprintf(1,'Wrote command for %s %d to cmdfile %s.\n',jobdesc,i,cmdfiles{i});
          tfSucc(i) = true;
        end
      end

    end

    function jobid = parseJobIDStatic(res,type)
      switch type,
        case DLBackEnd.Bsub,
          jobid = parseJobIDBsub(res);
        case DLBackEnd.Docker,
          jobid = parseJobIDDocker(res);
        otherwise
          error('Not implemented: %s',type);
      end
    end

    function jobid = parseJobIDBsub(res)
      PAT = 'Job <(?<jobid>[0-9]+)>';
      stoks = regexp(res,PAT,'names');
      if ~isempty(stoks)
        jobid = str2double(stoks.jobid);
      else
        jobid = nan;
        warning('Could not parse job id from:\n%s\',res);
      end
    end

    function jobID = parseJobIDDocker(res)
      res = regexp(res,'\n','split');
      res = regexp(res,'^[0-9a-f]+$','once','match');
      l = cellfun(@numel,res);
      try
        res = res{find(l==64,1)};
        assert(~isempty(res));
        jobID = strtrim(res);
      catch ME,
        warning('Could not parse job id from:\n%s\',res);
        disp(getReport(ME));
        jobID = '';
      end
    end

    function cmd = wrapCommandConda(basecmd,varargin) %#ok<INUSD,STOUT> 
      error('Not implemented');
    end

    function cmd = wrapCommandAWS(basecmd,varargin) %#ok<STOUT,INUSD> 
      error('Not implemented');
    end

    function cmdout = wrapCommandSing(cmdin,varargin)

      DFLTBINDPATH = {
        '/groups'
        '/nrs'
        '/scratch'};
      [bindpath,singimg] = myparse(varargin,...
        'bindpath',DFLTBINDPATH,...
        'singimg',DeepTracker.SINGULARITY_IMG_PATH...
        );
      bindpath = cellfun(@(x)['"' x '"'],bindpath,'uni',0);      
      Bflags = [repmat({'-B'},1,numel(bindpath)); bindpath(:)'];
      Bflagsstr = sprintf('%s ',Bflags{:});
      esccmd = String.escapeQuotes(cmdin);
      cmdout = sprintf('singularity exec --nv %s "%s" bash -c "%s"',...
        Bflagsstr,singimg,esccmd);

    end

    function cmdout = wrapCommandBsub(cmdin,varargin)
      [nslots,gpuqueue,logfile,jobname] = myparse(varargin,...
        'nslots',DeepTracker.default_jrcnslots_train,...
        'gpuqueue',DeepTracker.default_jrcgpuqueue,...
        'logfile','/dev/null',...
        'jobname','');
      esccmd = String.escapeQuotes(cmdin);
      if isempty(jobname),
        jobnamestr = '';
      else
        jobnamestr = [' -J ',jobname];
      end
      cmdout = sprintf('bsub -n %d -gpu "num=1" -q %s -o "%s" -R"affinity[core(1)]"%s "%s"',...
        nslots,gpuqueue,logfile,jobnamestr,esccmd);
    end

    function cmdout = wrapCommandSSH(remotecmd,varargin)

      [host,prefix,sshoptions,timeout,extraprefix] = myparse(varargin,...
        'host',DLBackEndClass.jrchost,...
        'prefix',DLBackEndClass.jrcprefix,...
        'sshoptions','-o "StrictHostKeyChecking no" -t',...
        'timeout',[],...
        'extraprefix','');

      if ~isempty(extraprefix),
        prefix = [prefix '; ' extraprefix];
      end

      if ~isempty(prefix),
        remotecmd = [prefix,'; ',remotecmd];
      end

      remotecmd = String.escapeQuotes(remotecmd);

      if ~isempty(timeout),
        sshoptions1 = ['-o "ConnectTimeout ',num2str(timeout),'"'];
        if ~ischar(sshoptions) || isempty(sshoptions),
          sshoptions = sshoptions1;
        else
          sshoptions = [sshoptions,' ',sshoptions1];
        end
      end
      if ~ischar(sshoptions) || isempty(sshoptions),
        sshcmd = 'ssh';
      else
        sshcmd = ['ssh ',sshoptions];
      end

      cmdout = sprintf('%s %s "%s"',sshcmd,host,remotecmd);

    end

    function cmd = wrapBaseCommandBsub(basecmd,varargin)

      [singargs,bsubargs,sshargs] = myparse(varargin,'singargs',{},'bsubargs',{},'sshargs',{});
      cmd = DLBackEndClass.wrapCommandSing(basecmd,singargs{:});
      cmd = DLBackEndClass.wrapCommandBsub(cmd,bsubargs{:});

      % already on cluster?
      tfOnCluster = ~isempty(getenv('LSB_DJOB_NUMPROC'));
      if ~tfOnCluster,
        cmd = DLBackEndClass.wrapCommandSSH(cmd,sshargs{:});
      end

    end


    function [hfig,hedit] = createFigTestConfig(figname)
      hfig = dialog('Name',figname,'Color',[0,0,0],'WindowStyle','normal');
      hedit = uicontrol(hfig,'Style','edit','Units','normalized',...
        'Position',[.05,.05,.9,.9],'Enable','inactive','Min',0,'Max',10,...
        'HorizontalAlignment','left','BackgroundColor',[.1,.1,.1],...
        'ForegroundColor',[0,1,0]);
    end
    
    function [tfsucc,hedit] = testBsubConfig(cacheDir,varargin)
      tfsucc = false;
      [host] = myparse(varargin,'host',DLBackEndClass.jrchost);
      
      [hfig,hedit] = DLBackEndClass.createFigTestConfig('Test JRC Cluster Backend');
      hedit.String = {sprintf('%s: Testing JRC cluster backend...',datestr(now))};
      drawnow;
      
      % test that you can ping jrc host
      hedit.String{end+1} = ''; drawnow;
      hedit.String{end+1} = sprintf('** Testing that host %s can be reached...\n',host); drawnow;
      cmd = sprintf('ping -c 1 -W 10 %s',host);
      hedit.String{end+1} = cmd; drawnow;
      [status,result] = system(cmd);
      hedit.String{end+1} = result; drawnow;
      if status ~= 0,
        hedit.String{end+1} = 'FAILURE. Error with ping command.'; drawnow;
        return;
      end
      % tried to make this robust to mac output
      m = regexp(result,' (\d+) [^,]*received','tokens','once');
      if isempty(m),
        hedit.String{end+1} = 'FAILURE. Could not parse ping output.'; drawnow;
        return;
      end
      if str2double(m{1}) == 0,
        hedit.String{end+1} = sprintf('FAILURE. Could not ping %s:\n',host); drawnow;
        return;
      end
      hedit.String{end+1} = 'SUCCESS!'; drawnow;
      
      % test that we can connect to jrc host and access CacheDir on it
      
      hedit.String{end+1} = ''; drawnow;
      hedit.String{end+1} = sprintf('** Testing that we can do passwordless ssh to %s...',host); drawnow;
      touchfile = fullfile(cacheDir,sprintf('testBsub_test_%s.txt',datestr(now,'yyyymmddTHHMMSS.FFF')));
      
      remotecmd = sprintf('touch "%s"; if [ -e "%s" ]; then rm -f "%s" && echo "SUCCESS"; else echo "FAILURE"; fi;',touchfile,touchfile,touchfile);
      timeout = 20;
      cmd1 = DeepTracker.codeGenSSHGeneral(remotecmd,'host',host,'bg',false,'timeout',timeout);
      %cmd = sprintf('timeout 20 %s',cmd1);
      cmd = cmd1;
      hedit.String{end+1} = cmd; drawnow;
      [status,result] = system(cmd);
      hedit.String{end+1} = result; drawnow;
      if status ~= 0,
        hedit.String{end+1} = sprintf('ssh command timed out. This could be because passwordless ssh to %s has not been set up. Please see APT wiki for more details.',host); drawnow;
        return;
      end
      issuccess = contains(result,'SUCCESS');
      isfailure = contains(result,'FAILURE');
      if issuccess && ~isfailure,
        hedit.String{end+1} = 'SUCCESS!'; drawnow;
      elseif ~issuccess && isfailure,
        hedit.String{end+1} = sprintf('FAILURE. Could not create file in CacheDir %s:',cacheDir); drawnow;
        return;
      else
        hedit.String{end+1} = 'FAILURE. ssh test failed.'; drawnow;
        return;
      end
      
      % test that we can run bjobs
      hedit.String{end+1} = '** Testing that we can interact with the cluster...'; drawnow;
      remotecmd = 'bjobs';
      cmd = DeepTracker.codeGenSSHGeneral(remotecmd,'host',host);
      hedit.String{end+1} = cmd; drawnow;
      [status,result] = system(cmd);
      hedit.String{end+1} = result; drawnow;
      if status ~= 0,
        hedit.String{end+1} = sprintf('Error running bjobs on %s',host); drawnow;
        return;
      end
      hedit.String{end+1} = 'SUCCESS!';
      hedit.String{end+1} = '';
      hedit.String{end+1} = 'All tests passed. JRC Backend should work for you.'; drawnow;
      
      tfsucc = true;
    end
    
  end
  
  methods % Bsub

    function aptroot = bsubSetRootUpdateRepo(obj,cacheDir,varargin)
      copyptw = myparse(varargin,...
        'copyptw',true ...
      );
      
      if obj.deepnetrunlocal
        aptroot = APT.Root;
      else
        DeepTracker.cloneJRCRepoIfNec(cacheDir);
        DeepTracker.updateAPTRepoExecJRC(cacheDir);
        aptroot = [cacheDir '/APT'];
      end
      if copyptw
        DeepTracker.cpupdatePTWfromJRCProdExec(aptroot);
      end
      obj.bsubaptroot = aptroot;
    end

    function bsubPretrack(obj,cacheDir)
      obj.bsubSetRootUpdateRepo(cacheDir);
    end
    
  end
  
  methods % Docker

    function s = dockercmd(obj)
      if isempty(obj.dockerremotehost)
        s = 'docker';
      else
        s = sprintf('ssh -t %s docker',obj.dockerremotehost);
      end
    end

    % KB 20191219: moved this to not be a static function so that we could
    % use this object's dockerremotehost
    function [tfsucc,clientver,clientapiver] = getDockerVers(obj)
      % Run docker cli to get docker versions
      %
      % tfsucc: true if docker cli command successful
      % clientver: if tfsucc, char containing client version; indeterminate otherwise
      % clientapiver: if tfsucc, char containing client apiversion; indeterminate otherwise
      
%       if isempty(obj.dockerremotehost),
%         dockercmd = 'docker';
%         dockercmdend = '';
%         filequote = '"';
%       else
%         dockercmd = sprintf('ssh -t %s "docker',obj.dockerremotehost);
%         dockercmdend = '"';
%         filequote = '\"';
%       end
%       
      FMTSPEC = '{{.Client.Version}}#{{.Client.DefaultAPIVersion}}';
      cmd = sprintf('docker version --format ''%s''%s',FMTSPEC);
      if ~isempty(obj.dockerremotehost),
        cmd = DLBackEndClass.wrapCommandSSH(cmd,'host',obj.dockerremotehost);
      end

      tfsucc = false;
      clientver = '';
      clientapiver = '';
        
      [st,res] = system(cmd);
      if st~=0
        return;
      end
      
      res = regexp(res,'\n','split'); % in case of ssh
      for i = 1:numel(res),
        res1 = res{i};
        res1 = strtrim(res1);
        toks = regexp(res1,'#','split');
        if numel(toks)~=2
          continue;
        end
      
        tfsucc = true;
        clientver = toks{1};
        clientapiver = toks{2};
        break;
      end
    end
    
    function filequote = getFileQuoteDockerCodeGen(obj) 
      % get filequote to use with codeGenDockerGeneral      
      if isempty(obj.dockerremotehost)
        % local Docker run
        filequote = '"';
      else
        filequote = '\"';
      end
    end
    
    function codestr = codeGenDockerGeneral(obj,basecmd,varargin)
      % Take a base command and run it in a docker img
      %
      % basecmd: currently assumed to have any filenames/paths protected by
      %   filequote as returned by obj.getFileQuoteDockerCodeGen
      
      DFLTBINDPATH = {};
      [containerName,bindpath,bindMntLocInContainer,dockerimg,isgpu,gpuid,tfDetach,...
        tty,shmSize] = ...
        myparse(varargin,...
        'containername','',...
        'bindpath',DFLTBINDPATH,... % paths on local filesystem that must be mounted/bound within container
        'binbMntLocInContainer','/mnt', ... % mount loc for 'external' filesys, needed if ispc+linux dockerim
        'dockerimg',obj.dockerimgfull,... % use :latest_cpu for CPU tracking
        'isgpu',true,... % set to false for CPU-only
        'gpuid',0,... % used if isgpu
        'detach',true, ...
        'tty',false,...
        'shmsize',[] ... optional
        );
      assert(~isempty(containerName));
      
      aptdeepnet = APT.getpathdl;
      
      tfWinAppLnxContainer = ispc;
      if tfWinAppLnxContainer
        % 1. Special treatment for bindpath. src are windows paths, dst are
        % linux paths inside /mnt.
        % 2. basecmd massage. All paths in basecmd will be windows paths;
        % these need to be replaced with the container paths under /mnt.
        srcbindpath = bindpath;
        dstbindpath = cellfun(...
          @(x,y)DeepTracker.codeGenPathUpdateWin2LnxContainer(x,bindMntLocInContainer),...
          srcbindpath,'uni',0);
        mountArgs = cellfun(@(x,y)sprintf('--mount type=bind,src=%s,dst=%s',x,y),...
          srcbindpath,dstbindpath,'uni',0);
        deepnetrootContainer = ...
          DeepTracker.codeGenPathUpdateWin2LnxContainer(aptdeepnet,bindMntLocInContainer);
        userArgs = {};
      else
        mountArgsFcn = @(x)sprintf('--mount "type=bind,src=%s,dst=%s"',x,x);
        % Can use raw bindpaths here; already in single-quotes, addnl
        % quotes unnec
        mountArgs = cellfun(mountArgsFcn,bindpath,'uni',0);
        deepnetrootContainer = aptdeepnet;
        userArgs = {'--user' '$(id -u):$(id -g)'};
      end
      
      if isgpu
        %nvidiaArgs = {'--runtime nvidia'};
        gpuArgs = {'--gpus' 'all'};
        cudaEnv = sprintf('export CUDA_DEVICE_ORDER=PCI_BUS_ID; export CUDA_VISIBLE_DEVICES=%d;',gpuid);
      else
        gpuArgs = cell(1,0);
        cudaEnv = 'export CUDA_VISIBLE_DEVICES=;'; 
        % MK 20220411 We need to explicitly set devices for pytorch when not using GPUS
      end
      
      homedir = getenv('HOME');
      user = getenv('USER');
      
      dockerApiVerExport = sprintf('export DOCKER_API_VERSION=%s;',obj.dockerapiver);

      dockercmd = sprintf('%s docker',dockerApiVerExport);

      if ~isempty(obj.dockerremotehost) && tfWinAppLnxContainer,
          error('Docker execution on remote host currently unsupported on Windows.');
          % Might work fine, maybe issue with double-quotes
      end
      
      if tfDetach,
        detachstr = '-d';
      else
        if tty
          detachstr = '-it';
        else
          detachstr = '-i';
        end        
      end
      
      otherargs = cell(0,1);
      if ~isempty(shmSize)
        otherargs{end+1,1} = sprintf('--shm-size=%dG',shmSize);
      end

      codestr = [
        {
        dockercmd
        'run'
        detachstr
        sprintf('--name %s',containerName);
        '--rm'
        '--ipc=host'
        '--network host'
        };
        mountArgs(:);
        gpuArgs(:);
        userArgs(:);
        otherargs(:);
        {
        '-w'
        ['"' deepnetrootContainer '"']
        '-e'
        ['USER=' user]
        dockerimg
        }
        ];
      bashcmd = sprintf('export HOME="%s"; %s cd "%s"; %s',...
        homedir,cudaEnv,deepnetrootContainer,basecmd);
      escbashcmd = sprintf('bash -c "%s"',String.escapeQuotes(bashcmd));
      codestr{end+1} = escbashcmd;      
      codestr = sprintf('%s ',codestr{:});
      codestr = codestr(1:end-1);
      if ~isempty(obj.dockerremotehost),
        codestr = DLBackEndClass.wrapCommandSSH(codestr,'host',obj.dockerremotehost);
      end
    end
    
    function [tfsucc,hedit] = testDockerConfig(obj)
      tfsucc = false;

      [hfig,hedit] = DLBackEndClass.createFigTestConfig('Test Docker Configuration');      
      hedit.String = {sprintf('%s: Testing Docker Configuration...',datestr(now))}; 
      drawnow;
      
      % docker hello world
      hedit.String{end+1} = ''; drawnow;
      hedit.String{end+1} = '** Testing docker hello-world...'; drawnow;
      
      if isempty(obj.dockerremotehost),
        dockercmd = 'docker';
        dockercmdend = '';
      else
        dockercmd = sprintf('ssh -t %s "docker',obj.dockerremotehost);
        dockercmdend = '"';
      end      
      cmd = sprintf('%s run hello-world%s',dockercmd,dockercmdend);
      fprintf(1,'%s\n',cmd);
      hedit.String{end+1} = cmd; drawnow;
      [st,res] = system(cmd);
      reslines = splitlines(res);
      reslinesdisp = reslines(1:min(4,end));
      hedit.String = [hedit.String; reslinesdisp(:)];
      if st~=0
        hedit.String{end+1} = 'FAILURE. Error with docker run command.'; drawnow;
        return;
      end
      hedit.String{end+1} = 'SUCCESS!'; drawnow;
      
      % docker (api) version
      hedit.String{end+1} = ''; drawnow;
      hedit.String{end+1} = '** Checking docker API version...'; drawnow;
      
      [tfsucc,clientver,clientapiver] = obj.getDockerVers;
      if ~tfsucc        
        hedit.String{end+1} = 'FAILURE. Failed to ascertain docker API version.'; drawnow;
        return;
      end
      
      tfsucc = false;
      % In this conditional we assume the apiver numbering scheme continues
      % like '1.39', '1.40', ... 
      if ~(str2double(clientapiver)>=str2double(obj.dockerapiver))          
        hedit.String{end+1} = ...
          sprintf('FAILURE. Docker API version %s does not meet required minimum of %s.',...
            clientapiver,obj.dockerapiver);
        drawnow;
        return;
      end        
      succstr = sprintf('SUCCESS! Your Docker API version is %s.',clientapiver);
      hedit.String{end+1} = succstr; drawnow;      
      
      % APT hello
      hedit.String{end+1} = ''; drawnow;
      hedit.String{end+1} = '** Testing APT deepnet library...'; 
      hedit.String{end+1} = '   (This can take some time the first time the docker image is pulled)'; 
      drawnow;
      deepnetroot = [APT.Root '/deepnet'];
      homedir = getenv('HOME');
      %deepnetrootguard = [filequote deepnetroot filequote];
      basecmd = 'python APT_interface.py lbl test hello';
      cmd = obj.codeGenDockerGeneral(basecmd,...
        'containername','containerTest',...
        'detach',false,...
        'bindpath',{deepnetroot,homedir});
      hedit.String{end+1} = cmd;
      RUNAPTHELLO = 1;
      if RUNAPTHELLO % AL: this may not work property on a multi-GPU machine with some GPUs in use
        %fprintf(1,'%s\n',cmd);
        %hedit.String{end+1} = cmd; drawnow;
        [st,res] = system(cmd);
        reslines = splitlines(res);
        reslinesdisp = reslines(1:min(4,end));
        hedit.String = [hedit.String; reslinesdisp(:)];
        if st~=0
          hedit.String{end+1} = 'FAILURE. Error with APT deepnet command.'; drawnow;
          return;
        end
        hedit.String{end+1} = 'SUCCESS!'; drawnow;
      end
      
      % free GPUs
      hedit.String{end+1} = ''; drawnow;
      hedit.String{end+1} = '** Looking for free GPUs ...'; drawnow;
      [gpuid,freemem,gpuifo] = obj.getFreeGPUs(1,'verbose',true);
      if isempty(gpuid)
        hedit.String{end+1} = 'WARNING. Could not find free GPUs. APT will run SLOWLY on CPU.'; drawnow;
      else
        hedit.String{end+1} = 'SUCCESS! Found available GPUs.'; drawnow;
      end
      
      hedit.String{end+1} = '';
      hedit.String{end+1} = 'All tests passed. Docker Backend should work for you.'; drawnow;
      
      tfsucc = true;      
    end
    
  end
  
  methods % Conda
    
    function [tfsucc,hedit] = testCondaConfig(obj)
      tfsucc = false;

      [hfig,hedit] = DLBackEndClass.createFigTestConfig('Test Conda Configuration');      
      hedit.String = {sprintf('%s: Testing Conda Configuration...',datestr(now))}; 
      drawnow;

      % make sure conda is installed
      hedit.String{end+1} = ''; drawnow;
      hedit.String{end+1} = '** Checking for conda...'; drawnow;
      cmd = 'conda -V';
      hedit.String{end+1} = cmd; drawnow;
      [st,res] = system(cmd);
      reslines = splitlines(res);
      if st~=0
        hedit.String{end+1} = sprintf('FAILURE. Error with ''%s''. Make sure you have installed conda and added it to your PATH.',cmd); drawnow;
        return;
      end
      hedit.String{end+1} = 'SUCCESS!'; drawnow;


      % activate APT
      hedit.String{end+1} = ''; drawnow;
      hedit.String{end+1} = '** Testing activate APT...'; drawnow;

      cmd = 'activate APT';
      %fprintf(1,'%s\n',cmd);
      hedit.String{end+1} = cmd; drawnow;
      [st,res] = system(cmd);
      reslines = splitlines(res);
      %reslinesdisp = reslines(1:min(4,end));
      %hedit.String = [hedit.String; reslinesdisp(:)];
      if st~=0
        hedit.String{end+1} = sprintf('FAILURE. Error with ''%s''. Make sure you have created the conda environment APT',cmd); drawnow;
        return;
      end
      hedit.String{end+1} = 'SUCCESS!'; drawnow;
        
%       TODO Mar2020: Consider adding APT hello
%       hedit.String{end+1} = ''; drawnow;
%       hedit.String{end+1} = '** Testing APT deepnet library...'; drawnow;
%       deepnetroot = [APT.Root '/deepnet'];
%       %deepnetrootguard = [filequote deepnetroot filequote];
%       basecmd = 'python APT_interface.py lbl test hello';
%       cmd = obj.codeGenDockerGeneral(basecmd,'containerTest',...
%         'detach',false,...
%         'bindpath',{deepnetroot});      
%       RUNAPTHELLO = 1;
%       if RUNAPTHELLO % AL: this may not work property on a multi-GPU machine with some GPUs in use
%         %fprintf(1,'%s\n',cmd);
%         %hedit.String{end+1} = cmd; drawnow;
%         [st,res] = system(cmd);
%         reslines = splitlines(res);
%         reslinesdisp = reslines(1:min(4,end));
%         hedit.String = [hedit.String; reslinesdisp(:)];
%         if st~=0
%           hedit.String{end+1} = 'FAILURE. Error with APT deepnet command.'; drawnow;
%           return;
%         end
%         hedit.String{end+1} = 'SUCCESS!'; drawnow;
%       end
      
      % free GPUs
      hedit.String{end+1} = ''; drawnow;
      hedit.String{end+1} = '** Looking for free GPUs ...'; drawnow;
      [gpuid,freemem,gpuifo] = obj.getFreeGPUs(1,'verbose',true);
      if isempty(gpuid)
        hedit.String{end+1} = 'WARNING: Could not find free GPUs. APT will run SLOWLY on CPU.'; drawnow;
      else
        hedit.String{end+1} = sprintf('SUCCESS! Found available GPUs.'); drawnow;
      end
      
      hedit.String{end+1} = '';
      hedit.String{end+1} = 'All tests passed. Conda Backend should work for you.'; drawnow;
      
      tfsucc = true;      
    end
  end
  
  methods % AWS
    
    function [tfsucc,hedit] = testAWSConfig(obj,varargin)
      tfsucc = false;
      [hfig,hedit] = DLBackEndClass.createFigTestConfig('Test AWS Backend');
      hedit.String = {sprintf('%s: Testing AWS backend...',datestr(now))}; 
      drawnow;
      
      % test that ssh exists
      hedit.String{end+1} = sprintf('** Testing that ssh is available...'); drawnow;
      hedit.String{end+1} = ''; drawnow;
      if ispc,
        isssh = exist(APT.WINSSHCMD,'file') && exist(APT.WINSCPCMD,'file');
        if isssh,
          hedit.String{end+1} = sprintf('Found ssh at %s',APT.WINSSHCMD); 
          drawnow;
        else
          hedit.String{end+1} = sprintf('FAILURE. Did not find ssh in the expected location: %s.',APT.WINSSHCMD); 
          drawnow;
          return;
        end
      else
        cmd = 'which ssh';
        hedit.String{end+1} = cmd; drawnow;
        [status,result] = system(cmd);
        hedit.String{end+1} = result; drawnow;
        if status ~= 0,
          hedit.String{end+1} = 'FAILURE. Did not find ssh.'; drawnow;
          return;
        end
      end
      
      if ispc,
        hedit.String{end+1} = sprintf('\n** Testing that certUtil is installed...\n'); drawnow;
        cmd = 'where certUtil';
        hedit.String{end+1} = cmd; drawnow;
        [status,result] = system(cmd);
        hedit.String{end+1} = result; drawnow;
        if status ~= 0,
          hedit.String{end+1} = 'FAILURE. Did not find certUtil.'; drawnow;
          return;
        end
      end
      
      awsec2 = obj.awsec2;

      % test that AWS CLI is installed
      hedit.String{end+1} = sprintf('\n** Testing that AWS CLI is installed...\n'); drawnow;
      cmd = 'aws ec2 describe-regions --output table';
      hedit.String{end+1} = cmd; drawnow;
      [tfsucc,result] = awsec2.syscmd(cmd,'dispcmd',true);
      %[status,result] = system(cmd);
      hedit.String{end+1} = result; drawnow;
      if ~tfsucc % status ~= 0,
        hedit.String{end+1} = 'FAILURE. Error using the AWS CLI.'; drawnow;
        return;
      end

      % test that apt_dl security group has been created
      hedit.String{end+1} = sprintf('\n** Testing that apt_dl security group has been created...\n'); drawnow;
      cmd = 'aws ec2 describe-security-groups';
      hedit.String{end+1} = cmd; drawnow;
      [tfsucc,result] = awsec2.syscmd(cmd,'dispcmd',true,'isjsonout',true);
      %[status,result] = system(cmd);
      if tfsucc %status == 0,
        try
          result = jsondecode(result);
          if ismember('apt_dl',{result.SecurityGroups.GroupName}),
            hedit.String{end+1} = 'Found apt_dl security group.'; drawnow;
          else
            status = 1;
          end
        catch
          status = 1;
        end
        if status == 1,
          hedit.String{end+1} = 'FAILURE. Could not find the apt_dl security group.'; drawnow;
        end
      else
        hedit.String{end+1} = result; drawnow;
        hedit.String{end+1} = 'FAILURE. Error checking for apt_dl security group.'; drawnow;
        return;
      end
      
      % to do, could test launching an instance, or at least dry run

%       m = regexp(result,' (\d+) received, (\d+)% packet loss','tokens','once');
%       if isempty(m),
%         hedit.String{end+1} = 'FAILURE. Could not parse ping output.'; drawnow;
%         return;
%       end
%       if str2double(m{1}) == 0,
%         hedit.String{end+1} = sprintf('FAILURE. Could not ping %s:\n',host); drawnow;
%         return;
%       end
%       hedit.String{end+1} = 'SUCCESS!'; drawnow;
%       
%       % test that we can connect to jrc host and access CacheDir on it
%      
%       hedit.String{end+1} = ''; drawnow;
%       hedit.String{end+1} = sprintf('** Testing that we can do passwordless ssh to %s...',host); drawnow;
%       touchfile = fullfile(cacheDir,sprintf('testBsub_test_%s.txt',datestr(now,'yyyymmddTHHMMSS.FFF')));
%       
%       remotecmd = sprintf('touch %s; if [ -e %s ]; then rm -f %s && echo "SUCCESS"; else echo "FAILURE"; fi;',touchfile,touchfile,touchfile);
%       cmd1 = DeepTracker.codeGenSSHGeneral(remotecmd,'host',host,'bg',false);
%       cmd = sprintf('timeout 20 %s',cmd1);
%       hedit.String{end+1} = cmd; drawnow;
%       [status,result] = system(cmd);
%       hedit.String{end+1} = result; drawnow;
%       if status ~= 0,
%         hedit.String{end+1} = sprintf('ssh command timed out. This could be because passwordless ssh to %s has not been set up. Please see APT wiki for more details.',host); drawnow;
%         return;
%       end
%       issuccess = contains(result,'SUCCESS');
%       isfailure = contains(result,'FAILURE');
%       if issuccess && ~isfailure,
%         hedit.String{end+1} = 'SUCCESS!'; drawnow;
%       elseif ~issuccess && isfailure,
%         hedit.String{end+1} = sprintf('FAILURE. Could not create file in CacheDir %s:',cacheDir); drawnow;
%         return;
%       else
%         hedit.String{end+1} = 'FAILURE. ssh test failed.'; drawnow;
%         return;
%       end
%       
%       % test that we can run bjobs
%       hedit.String{end+1} = '** Testing that we can interact with the cluster...'; drawnow;
%       remotecmd = 'bjobs';
%       cmd = DeepTracker.codeGenSSHGeneral(remotecmd,'host',host);
%       hedit.String{end+1} = cmd; drawnow;
%       [status,result] = system(cmd);
%       hedit.String{end+1} = result; drawnow;
%       if status ~= 0,
%         hedit.String{end+1} = sprintf('Error running bjobs on %s',host); drawnow;
%         return;
%       end
      hedit.String{end+1} = 'SUCCESS!'; 
      hedit.String{end+1} = ''; 
      hedit.String{end+1} = 'All tests passed. AWS Backend should work for you.'; drawnow;
      
      tfsucc = true;      
    end
    
    function awsPretrack(obj,dmc,setstatusfn)
      setstatusfn('AWS Tracking: Uploading code and data...');
      
      obj.awsUpdateRepo();
      aws = obj.awsec2;
      if ~isempty(dmc) && dmc.isRemote,
        dmc.mirror2remoteAws(aws);
      end
      
      setstatusfn('Tracking...');      
    end
    
    function awsUpdateRepo(obj) % throws if fails
      if isempty(obj.awsgitbranch)
        args = {};
      else
        args = {'branch' obj.awsgitbranch};
      end
      cmdremote = DeepTracker.updateAPTRepoCmd('downloadpretrained',true,args{:});

      aws = obj.awsec2;      
      [tfsucc,res] = aws.cmdInstance(cmdremote,'dispcmd',true); %#ok<ASGLU>
      if tfsucc
        fprintf('Updated remote APT repo.\n\n');
      else
        error('Failed to update remote APT repo.');
      end
    end
    
  end
  
end
    