classdef DLBackEndClass < handle
  % APT Backends specify a physical machine/server where GPU code is run.
  % This class is intended to abstract away details particular to one backend or
  % another, so that calling code doesn't need to worry about such grubby
  % details.
  
  properties (Constant)
    minFreeMem = 9000  % in MiB
    defaultDockerImgTag = 'apt_20230427_tf211_pytorch113_ampere'
    defaultDockerImgRoot = 'bransonlabapt/apt_docker'
 
    jrchost = 'login1.int.janelia.org'
    jrcprefix = ''
    jrcprodrepo = '/groups/branson/bransonlab/apt/repo/prod'
    default_jrcgpuqueue = 'gpu_a100'
    default_jrcnslots_train = 4
    default_jrcnslots_track = 4

    default_conda_env = 'APT'
    default_singularity_image_path = '/groups/branson/bransonlab/apt/sif/apt_20230427_tf211_pytorch113_ampere.sif' 
    legacy_default_singularity_image_path = '/groups/branson/bransonlab/apt/sif/prod.sif'
    legacy_default_singularity_image_path_for_detect = '/groups/branson/bransonlab/apt/sif/det.sif'

    %default_docker_api_version = '1.40'
  end

  properties
    type  % scalar DLBackEnd

    % Used only for type==Bsub
    deepnetrunlocal = true
      % scalar logical. if true, bsub backend runs code in APT.Root/deepnet.
      % This path must be visible in the backend or else.
      % Applies only to bsub. Name should be eg 'bsubdeepnetrunlocal'
    bsubaptroot = []  % root of APT repo for bsub backend running     
    jrcsimplebindpaths = true  % whether to bind '/groups', '/nrs' for the Bsub/JRC backend
        
    % Used only for type==AWS
    awsec2  % a scalar AWSec2 object (present whether we need it or not)
    awsgitbranch  
      % Stores the branch name of APT to use when updating APT on the AWS EC2
      % instance.  This is never set in the APT codebase, as near as I can tell.
      % Likely used only for debugging?  -- ALT, 2024-03-07
    
    % Used only for type==Docker  
    %dockerapiver = DLBackEndClass.default_docker_api_version  % docker codegen will occur against this docker api ver
    dockerimgroot = DLBackEndClass.defaultDockerImgRoot
      % We have an instance prop for this to support running on older/custom
      % docker images.
    dockerimgtag = DLBackEndClass.defaultDockerImgTag
    dockerremotehost = ''
      % The docker backend can run the docker container on a remote host.
      % dockerremotehost will contain the DNS name of the remote host in this case.
      % But even in this case, as with local docker, we assume that the docker
      % container and the local host have the same filesystem paths to all the
      % training/tracking files we will access.  (Like if e.g. they're both managed
      % Linux boxes on the Janelia network.)

    gpuids = []  % for now used by docker/conda
%     dockercontainername = []  
%       % transient
%       % Also, seemingly never read -- ALT, 2024-03-07
    
    jrcAdditionalBsubArgs = ''  % Additional arguments to be passed to JRC bsub command, e.g. '-P scicompsoft'    
    jrcgpuqueue 
    jrcnslots 
    jrcnslotstrack

    condaEnv = DLBackEndClass.default_conda_env   % used only for Conda

    % We set these to the string 'invalid' so we can catch them in loadobj()
    % They are set properly in the constructor.
    singularity_image_path_ = '<invalid>'
    does_have_special_singularity_detection_image_path_ = '<invalid>'
    singularity_detection_image_path_ = '<invalid>'
  end

  properties (Transient)
    % The job registry.  These are protected in spirit.
    % These are jobs that can be spawned with a subsequent call to
    % spawnRegisteredJobs().
    training_syscmds_ = cell(0,1)
    training_cmdfiles_ = cell(0,1)
    training_logcmds_ = cell(0,1)
    tracking_syscmds_ = cell(0,1)
    tracking_cmdfiles_ = cell(0,1)
    tracking_logcmds_ = cell(0,1)

    % The job IDs.  These are protected, in spirit.
    % Each job id is represented as an old-style *string*.  What exactly they mean
    % depends on the backend.  For conda backend, the job id is the PGID of the
    % process group of the Python APT_interface.py invocation.  For Docker and
    % AWS, it's the Docker process ID.  For bsub, it's the LSF job number.
    training_jobids_ = cell(0,1)
    tracking_jobids_ = cell(0,1)

    % This is used to keep track of whether we need to release/delete resources on
    % delete()
    doesOwnResources_ = true  % is obj a copy, or the original
  end

  properties (Dependent)
    dockerimgfull % full docker img spec (with tag if specified)
    singularity_image_path
    singularity_detection_image_path
    isInAwsDebugMode
    isDMCRemote
    isDMCLocal
    localDMCRootDir
    remoteDMCRootDir
    awsInstanceID
    awsKeyName
    awsPEM
    awsInstanceType
  end
  
  methods
    function obj = DLBackEndClass(ty)
      if ~exist('ty', 'var') || isempty(ty) ,
        ty = DLBackEnd.Bsub ;
      end
      obj.type = ty ;
      % Set the singularity fields to valid values
      obj.singularity_image_path_ = DLBackEndClass.default_singularity_image_path ;
      obj.does_have_special_singularity_detection_image_path_ = false ;
      obj.singularity_detection_image_path_ = '' ;

      %MK 20250326. The jrc values weren't get initialized
      obj.jrcgpuqueue = DLBackEndClass.default_jrcgpuqueue;
      obj.jrcnslots = DLBackEndClass.default_jrcnslots_train;
      obj.jrcnslotstrack = DLBackEndClass.default_jrcnslots_track;

      % Just populate this now, whether or not we end up using it      
      obj.awsec2 = AWSec2() ;
    end
  end
  
  methods % Prop access
    function set.type(obj, raw_value)
      if ischar(raw_value) || isstring(raw_value) ,
        value = DLBackEndFromString(raw_value) ;
      elseif isa(raw_value, 'DLBackEnd') ,
        value = raw_value ;
      else
        error('Argument to DLBackEndClass::set.type() must be a row char array, a scalar string array, or a DLBackEnd') ;
      end      
      obj.type = value ;      
    end

    function v = get.dockerimgfull(obj)
      v = obj.dockerimgroot;
      if ~isempty(obj.dockerimgtag)
        v = [v ':' obj.dockerimgtag];
      end
    end

    function set.dockerimgfull(obj, new_value)
      % Check for crazy values
      if ischar(new_value) && ~isempty(new_value) ,
        % all is well
      else
        error('APT:invalidValue', 'Invalid value for the Docker image specification');
      end        
      % The full image spec should be of the form '<root>:<tag>' or just '<root>'
      % Parse the given string to find the parts
      parts = strsplit(new_value, ':');
      part_count = length(parts) ;      
      if part_count==0 ,
        error('APT:internalerror', 'Internal APT error.  Please notify the APT developers.');
      elseif part_count==1 ,
        root = parts{1} ;
        tag = '' ;
      elseif part_count==2 ,
        root = parts{1} ;
        tag = parts{2} ;
      else
        error('APT:invalidValue', '"%s" is a not valid value for the Docker image specification', new_value);
      end
      % Actually set the values
      obj.dockerimgroot = root ;
      obj.dockerimgtag = tag ;
    end    

    function result = get.singularity_detection_image_path(obj)
      if obj.does_have_special_singularity_detection_image_path_ ,
        result = obj.singularity_detection_image_path_ ;
      else
        result = obj.singularity_image_path_ ;
      end
    end

    function set.singularity_image_path(obj, new_value)
      if ischar(new_value) && exist(new_value, 'file') ,
        obj.singularity_image_path_ = new_value ;
        obj.does_have_special_singularity_detection_image_path_ = false ;
        obj.singularity_detection_image_path_ = '' ;
      else
        error('APT:invalidValue', 'Invalid value for the Singularity image path');
      end        
    end

    function result = get.singularity_image_path(obj)
      result = obj.singularity_image_path_ ;
    end

    function set.jrcAdditionalBsubArgs(obj, new_value)
      % Check for crazy values
      if ischar(new_value) ,
        % all is well
      else
        error('APT:invalidValue', 'Invalid value for the JRC addition bsub arguments');
      end        
      % Actually set the value
      obj.jrcAdditionalBsubArgs = new_value ;
    end    

    function set.condaEnv(obj, new_value)
      % Check for crazy values
      if ischar(new_value) && ~isempty(new_value) ,
        % all is well
      else
        error('APT:invalidValue', '"%s" is a not valid value for the conda environment', new_value);
      end        
      % Actually set the value
      obj.condaEnv = new_value ;
    end    
  end  % methods block
 
  methods
    function [return_code, stdouterr] = runBatchCommandOutsideContainer(obj, basecmd, varargin)
      % Run the basecmd using apt.syscmd(), after wrapping suitably for the type of
      % backend.  But as the name implies, commands are run outside the backend
      % container/environment.  For the AWS backend, this means commands are run
      % outside the Docker environment.  For the Bsub backend, commands are run
      % outside the Apptainer container.  For the Conda backend, commands are run
      % outside the conda environment (i.e. they are simply run).  For the Docker
      % backend, commands are run outside the Docker container (for local Docker,
      % this means they are simply run; for remote Docker, this means they are run
      % via ssh, but outside the Docker container).  This function blocks, and
      % doesn't return a process identifier of any kind.  Return values are like
      % those from system(): a numeric return code and a string containing any
      % command output. Note that any file names in the basecmd must refer to the
      % filenames on the *backend* filesystem (and within WSL if running on
      % Windows).
      switch obj.type,
        case DLBackEnd.AWS
          % For AWS backend, use the AWSec2 method of the same name
          [return_code, stdouterr] = obj.awsec2.runBatchCommandOutsideContainer(basecmd, varargin{:}) ;
        case DLBackEnd.Bsub,
          % For now, we assume Matlab frontend is running on a JRC cluster node,
          % which means the filesystem is local.
          command = basecmd ;
          [return_code, stdouterr] = apt.syscmd(command, 'failbehavior', 'silent', 'verbose', false, varargin{:}) ;
        case DLBackEnd.Conda
          command = basecmd ;
          [return_code, stdouterr] = apt.syscmd(command, 'failbehavior', 'silent', 'verbose', false, varargin{:}) ;
        case DLBackEnd.Docker
          % If docker host is remote, we assume all files we need to access are on the
          % same path on the remote host.
          command = basecmd ;
          [return_code, stdouterr] = apt.syscmd(command, 'failbehavior', 'silent', 'verbose', false, varargin{:}) ;
        otherwise
          error('Not implemented: %s',obj.type);
      end
        % Things passed in with varargin should overide things we set here
    end  % function

    function result = remoteMoviePathFromLocal(obj, localPath)
      % Convert a local movie path to the remote equivalent.
      % For non-AWS backends, this is the identity function.
      if isequal(obj.type, DLBackEnd.AWS) ,
        result = AWSec2.remoteMoviePathFromLocal(localPath) ;
      else
        result = localPath ;
      end
    end

    function result = remoteMoviePathsFromLocal(obj, localPathFromMovieIndex)
      % Convert a cell array of local movie paths to their remote equivalents.
      % For non-AWS backends, this is the identity function.
      if isequal(obj.type, DLBackEnd.AWS) ,
        result = AWSec2.remoteMoviePathsFromLocal(localPathFromMovieIndex) ;
      else
        result = localPathFromMovieIndex ;
      end
    end

    function uploadMovies(obj, localPathFromMovieIndex)
      % Upload movies to the backend, if necessary.
      if isequal(obj.type, DLBackEnd.AWS) ,
        obj.awsec2.uploadMovies(localPathFromMovieIndex) ;
      end
    end  % function

    % function uploadOrVerifySingleFile_(obj, localPath, remotePath, fileDescription)
    %   % Upload a single file.  Protected by convention.
    %   % Doesn't check to see if the backend type has a different filesystem.  That's
    %   % why outsiders shouldn't call it.
    %   localFileDirOutput = dir(localPath) ;
    %   localFileSizeInKibibytes = round(localFileDirOutput.bytes/2^10) ;
    %   % We just use scpUploadOrVerify which does not confirm the identity
    %   % of file if it already exists. These movie files should be
    %   % immutable once created and their naming (underneath timestamped
    %   % modelchainIDs etc) should be pretty/totally unique. 
    %   %
    %   % Only situation that might cause problems are augmentedtrains but
    %   % let's not worry about that for now.
    %   localFileName = localFileDirOutput.name ;
    %   fullFileDescription = sprintf('%s (%s), %d KiB', fileDescription, localFileName, localFileSizeInKibibytes) ;
    %   obj.scpUploadOrVerify(localPath, ...
    %                         remotePath, ...
    %                         fullFileDescription, ...
    %                         'destRelative',false) ;  % throws      
    % end  % function

    function delete(obj)
      if obj.doesOwnResources_ ,
        obj.killAndClearRegisteredJobs('track') ;
        obj.killAndClearRegisteredJobs('train') ;
        obj.stopEc2InstanceIfNeeded_() ;
      end
    end  % function    
  end  % methods

  methods
    function modernize(obj)
      % 20220728 Win/Conda migration to WSL2/Docker
      if obj.type==DLBackEnd.Conda
        if ispc() ,
          warningNoTrace(...
            ['Updating Windows backend from Conda -> Docker. ' ...
             'If you have not already, please see the documentation for Windows/WSL2 setup instructions.']);
          obj.type = DLBackEnd.Docker;
        else
          % Going to skip this warning.  Conda backend is fine.
          %warningNoTrace('Current backend is Conda.  This is only intended for developers.  Be careful.');
        end
      end

      % 20211101 turn on by default
      obj.jrcsimplebindpaths = 1;
      
      % In modern versions, we always have a .awsec2, whether we need it or not
      if isempty(obj.awsec2) ,
        obj.awsec2 = AWSec2() ;
      end

      % If these JRC-backend-related things are empty, warn that we're using default values
      if isempty(obj.jrcgpuqueue) || strcmp(obj.jrcgpuqueue,'gpu_any') || strcmp(obj.jrcgpuqueue,'gpu_tesla') || startsWith(obj.jrcgpuqueue,'gpu_rtx') ,
        obj.jrcgpuqueue = DLBackEndClass.default_jrcgpuqueue ;
        warningNoTrace('Updating JRC GPU cluster queue to ''%s''.', DLBackEndClass.default_jrcgpuqueue) ;
      end
      if isempty(obj.jrcnslots) ,
        obj.jrcnslots = DLBackEndClass.default_jrcnslots_train ;
        warningNoTrace('Updating JRC GPU cluster training slot count to %d.', DLBackEndClass.default_jrcnslots_train) ;
      end
      if isempty(obj.jrcnslotstrack) ,
        obj.jrcnslotstrack = DLBackEndClass.default_jrcnslots_track ;
        warningNoTrace('Updating JRC GPU cluster tracking slot count to %d.', DLBackEndClass.default_jrcnslots_track) ;
      end      
    end  % function
    
    function [isReady, reasonNotReady] = ensureIsRunning(obj)
      % If the backend is not 'running', tell it to start, and wait for it to be
      % fully started.  On return, isRunning reflects whether this worked.  If
      % isRunning is false, reasonNotRunning is a string that says something about
      % what went wrong.  This is essentially a no-op all but the AWS backends.  For
      % the AWS backend, it actually does (try to) make sure the AWS EC2 instance is
      % running.

      if obj.type==DLBackEnd.AWS
        [isReady, reasonNotReady] = obj.awsec2.ensureIsRunning() ;
      elseif obj.type==DLBackEnd.Conda ,
        if ispc() ,
          isReady = false ;
          reasonNotReady = 'Conda backend is not supported on Windows.' ;
        else
          isReady = true ;
          reasonNotReady = '' ;
        end
      else
        isReady = true;
        reasonNotReady = '';
      end
    end  % method
    
    function s = prettyName(obj)
      switch obj.type,
        case DLBackEnd.AWS,
          s = 'Docker';
        case DLBackEnd.Bsub,
          s = 'JRC Cluster';
        case DLBackEnd.Conda,
          s = 'Conda';
        case DLBackEnd.Docker,
          s = 'Docker';
        otherwise
          error('Unknown backend type') ;
      end
    end

    function v = isGpuLocal(obj)
      % Whether the Python training/tracking code will run on a GPU in the same
      % machine as the Matlab frontend.  This is true for the Conda backend, false
      % for the Bsub (i.e. Janelia LSF) backend and the AWS backend.  This is true
      % for the Docker backend, unless a Docker remote host has been specified, in
      % which case it is false.
      is_docker_and_local = isequal(obj.type, DLBackEnd.Docker) && isempty(obj.dockerremotehost) ;
      v = is_docker_and_local || isequal(obj.type,DLBackEnd.Conda) ;
    end
    
    function v = isGpuRemote(obj)
      v = ~obj.isGpuLocal(obj) ;
    end
    
    function v = isFilesystemRemote(obj)
      % The conda and bsub (i.e. Janelia LSF) and Docker backends share (mostly) the
      % same filesystem as the Matlab process.  AWS does not.
      % Note that for bsub and remote docker backends, we return true, but we're
      % assuming that all the files used by APT are on the part of the filesystem
      % that is actually the same between the frontend and the backend.
      v = isequal(obj.type,DLBackEnd.AWS) ;
    end
    
    function v = isFilesystemLocal(obj)
      v = ~obj.isFilesystemRemote() ;
    end
    
    function [gpuid, freemem, gpuInfo] = getFreeGPUs(obj, nrequest, varargin)
      % Get free gpus subject to minFreeMem constraint (see optional PVs)
      %
      % This sets .gpuids
      % 
      % gpuid: [ngpu] where ngpu<=nrequest, depending on if enough GPUs are available
      % freemem: [ngpu] etc
      % gpuInfo: scalar struct

      [~, minFreeMem, condaEnv, verbose] = ...
        myparse(varargin,...
                'dockerimg',obj.dockerimgfull,...
                'minfreemem',obj.minFreeMem,...
                'condaEnv',obj.condaEnv,...
                'verbose',0) ;  %#ok<PROPLC>
      
      gpuid = [];
      freemem = 0;
      gpuInfo = [];
      aptdeepnetpath = APT.getpathdl() ;
      
      switch obj.type,
        case DLBackEnd.Docker
          basecmd = 'echo START; python parse_nvidia_smi.py; echo END';
          bindpath = {aptdeepnetpath}; % don't use guarded
          codestr = wrapCommandDocker(basecmd,...
                                      'dockerimg',obj.dockerimgfull,...
                                      'containername','aptTestContainer',...
                                      'bindpath',bindpath,...
                                      'detach',false);
          if verbose
            fprintf(1,'%s\n',codestr);
          end
          [st,res] = apt.syscmd(codestr);
          if st ~= 0,
            warning('Error getting GPU info: %s\n%s',res,codestr);
            return;
          end
        case DLBackEnd.Conda
          scriptpath = fullfile(aptdeepnetpath, 'parse_nvidia_smi.py') ;
          basecmd = sprintf('echo START && python %s && echo END', scriptpath);
          codestr = wrapCommandConda(basecmd, 'condaEnv', condaEnv) ;
          [st,res] = apt.syscmd(codestr) ;  % wrapCommandConda 
          if st ~= 0,
            warning('Error getting GPU info: %s',res);
            return
          end
        case DLBackEnd.Bsub
          % We basically want to skip all the checks etc, so return values that will
          % make that happen.
          gpuid = 1 ;
          freemem = inf ;
          gpuInfo = [] ;
          obj.gpuids = 1 ;
          return
        case DLBackEnd.AWS
          % We basically want to skip all the checks etc, so return values that will
          % make that happen.
          gpuid = 1 ;
          freemem = inf ;
          gpuInfo = [] ;
          obj.gpuids = 1 ;
          return
        otherwise
          error('APT:internalError', 'Internal error: backend type %s not recognized', char(obj.type)) ;
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

      freemem = freemem(1:ngpu);
      gpuid = gpuid(1:ngpu);
      
      ngpureturn = min(ngpu,nrequest);
      gpuid = gpuid(1:ngpureturn);

      freemem = freemem(1:ngpureturn);
      
      obj.gpuids = gpuid;
    end
    
    function r = aptSourceDirRoot(obj)
      switch obj.type
        case DLBackEnd.Bsub
          r = obj.bsubaptroot;
        case DLBackEnd.AWS
          r = AWSec2.remoteAPTSourceRootDir ;
        case DLBackEnd.Docker
          r = APT.Root;          
        case DLBackEnd.Conda
          r = APT.Root;          
      end
    end

    % function r = getAPTDeepnetRoot(obj)
    %   r = [obj.aptSourceDirRoot '/deepnet'];
    % end
        
    function tfSucc = writeCmdToFile(obj, syscmds, cmdfiles, jobdesc)  % const method
      % Write each syscmds{i} to each cmdfiles{i}, on the filesystem where the
      % commands will be executed.
      if nargin < 4,
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
        syscmd = syscmds{i} ;
        cmdfile = cmdfiles{i} ;
        syscmdWithNewline = sprintf('%s\n', syscmd) ;
        [didSucceed, errorMessage] = obj.writeStringToFile(cmdfile, syscmdWithNewline) ;
        tfSucc(i) = didSucceed ;
        if didSucceed ,
          fprintf('Wrote command for %s %d to cmdfile %s.\n',jobdesc,i,cmdfile);
        else
          warningNoTrace(errorMessage);
        end
      end  % for
    end  % function
  end  % methods block
  
  methods % Bsub
    function aptroot = bsubSetRootUpdateRepo_(obj)
      aptroot = apt.bsubSetRootUpdateRepo(obj.deepnetrunlocal) ;
      obj.bsubaptroot = aptroot;
    end
  end  % methods
  
  methods % Docker
    % KB 20191219: moved this to not be a static function so that we could
    % use this object's dockerremotehost
    function [tfsucc,clientver,clientapiver] = getDockerVers(obj)
      % Run docker cli to get docker versions
      %
      % tfsucc: true if docker cli command successful
      % clientver: if tfsucc, char containing client version; indeterminate otherwise
      % clientapiver: if tfsucc, char containing client apiversion; indeterminate otherwise
      
      dockercmd = apt.dockercmd();      
      fmtspec = '{{.Client.Version}}#{{.Client.DefaultAPIVersion}}';
      cmd = sprintf('%s version --format "%s"',dockercmd,fmtspec);
      if ~isempty(obj.dockerremotehost),
        cmd = wrapCommandSSH(cmd,'host',obj.dockerremotehost);
      end
      
      tfsucc = false;
      clientver = '';
      clientapiver = '';
        
      [st,res] = apt.syscmd(cmd);
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
    end  % function
  end  % methods
  
  methods % AWS
    % function checkConnection(obj)  
    %   % Errors if connection to backend is ok.  Otherwise returns nothing.
    %   if isequal(obj.type, DLBackEnd.AWS) ,
    %     aws = obj.awsec2;
    %     aws.checkInstanceRunning() ;
    %   end
    % end

    function scpUploadOrVerify(obj, varargin)
      if isequal(obj.type, DLBackEnd.AWS) ,
        aws = obj.awsec2;
        aws.scpUploadOrVerify(varargin{:}) ;
      end      
    end

    % function rsyncUpload(obj, src, dest)
    %   if isequal(obj.type, DLBackEnd.AWS) ,
    %     aws = obj.awsec2 ;
    %     aws.rsyncUpload(src, dest) ;
    %   end      
    % end
    
  end  % public methods block

  methods
    function [didsucceed, msg] = mkdir(obj, dir_name)
      % Create the named directory, either locally or remotely, depending on the
      % backend type.      
      if obj.type == DLBackEnd.AWS ,
        [didsucceed, msg] = obj.awsec2.mkdir(dir_name) ;
      else
        quoted_dirloc = escape_string_for_bash(dir_name) ;
        base_command = sprintf('mkdir -p %s', quoted_dirloc) ;
        [status, msg] = obj.runBatchCommandOutsideContainer(base_command) ;
        didsucceed = (status==0) ;
      end
    end  % function

    function [didsucceed, msg] = deleteFile(obj, file_name)
      % Delete the named file, either locally or remotely, depending on the
      % backend type.
      quoted_file_name = escape_string_for_bash(file_name) ;
      base_command = sprintf('rm %s', quoted_file_name) ;
      [status, msg] = obj.runBatchCommandOutsideContainer(base_command) ;
      didsucceed = (status==0) ;
    end

    % function [doesexist, msg] = exist(obj, file_name, file_type)
    %   % Check whether the named file/dir exists, either locally or remotely,
    %   % depending on the backend type.
    %   if ~exist('file_type', 'var') ,
    %     file_type = '' ;
    %   end
    %   if strcmpi(file_type, 'dir') ,
    %     option = '-d' ;
    %   elseif strcmpi(file_type, 'file') ,
    %     option = '-f' ;
    %   else
    %     option = '-e' ;
    %   end
    %   quoted_file_name = escape_string_for_bash(file_name) ;
    %   base_command = sprintf('test %s %s', option, quoted_file_name) ;
    %   [status, msg] = obj.runBatchCommandOutsideContainer(base_command) ;
    %   doesexist = (status==0) ;
    % end

    function [didSucceed, errorMessage] = writeStringToFile(obj, filename, str)
      % Write the given string to a file, overrwriting any previous contents.
      % For remote backends, uses a single "ssh echo $string > $filename" to do
      % this, so limited to strings of ~10^5 bytes.
      if obj.isFilesystemLocal() ,
        % Filesystem is local
        try
          fo = file_object(filename, 'w') ;
        catch me 
          if strcmp(me.identifier, 'file_object:unable_to_open') ,
            didSucceed = false ;
            errorMessage = sprintf('Could not open file %s for writing: %s', filename, me.message) ;
            return
          else
            rethrow(me) ;
          end
        end  % try-catch
        fprintf(fo, '%s', str) ;
        fclose(fo) ;
      else
        % Filesystem is remote
        if strlength(str) > 100000 ,
          didSucceed = false ;
          errorMessage = ...
            sprintf(['Could not write to file %s: ' ...
                     'Current implementation of DLBackEndClass.writeStringToFile() only supports strings of length 100,000 or less'], ...
                    filename) ;
          return
        end          
        quoted_file_name = escape_string_for_bash(filename) ;
        quoted_str = escape_string_for_bash(str) ;
        base_command = sprintf('echo %s > %s', quoted_str, quoted_file_name) ;
        [status, msg] = obj.runBatchCommandOutsideContainer(base_command) ;
        if status ~= 0 ,
          didSucceed = false ;
          errorMessage = sprintf('Something went wrong while writing to backend file %s: %s',filename,msg);
          return
        end
      end
      didSucceed = true ;
      errorMessage = '' ;
    end  % function    

    function updateRepo(obj)
      % Update the APT repo on the backend.  While we're at it, make sure the
      % pretrained weights are downloaded.  The method formerly known as
      % setupForTrainingOrTracking().
      % localCacheDir should be e.g. /home/joeuser/.apt/tp662830c8_246a_49c6_816c_470db4ecd950
      % localCacheDir is not currently used, but will be needed to get the JRC
      % backend working properly for AD-linked Linux workstations.
      switch obj.type
        case DLBackEnd.Bsub ,
          obj.bsubSetRootUpdateRepo_();
        case {DLBackEnd.Conda, DLBackEnd.Docker} ,
          aptroot = APT.Root;
          apt.downloadPretrainedWeights('aptroot', aptroot) ;
        case DLBackEnd.AWS ,
          obj.awsec2.updateRepo() ;
        otherwise
          error('Unknown backend type') ;
      end
    end  % function    

    function result = getLocalMoviePathFromRemote(obj, queryRemotePath)
      if obj.type == DLBackEnd.AWS ,
        result = obj.awsec2.getLocalMoviePathFromRemote(queryRemotePath) ;
      else
        result = queryRemotePath ;
      end
    end  % function

    function result = getRemoteMoviePathFromLocal(obj, queryLocalPath)
      if obj.type == DLBackEnd.AWS ,
        result = obj.awsec2.getRemoteMoviePathFromLocal(queryLocalPath) ;
      else
        result = queryLocalPath ;
      end
    end  % function
  end  % methods

  % These next two methods allow access to private and protected variables,
  % intended to be used for encoding/decoding.  The trailing underscore is there
  % to remind you that these methods are only intended for "special case" uses.
  methods
    function result = get_property_value_(obj, name)
      result = obj.(name) ;
    end  % function
    
    function set_property_value_(obj, name, value)
      obj.(name) = value ;
    end  % function
  end

  methods (Static)
    function obj = loadobj(larva)
      % We implement this to provide backwards-compatibility with older .mat files
      if isstruct(larva) ,
        obj = DLBackEndClass() ;
        field_names = fieldnames(larva) ;
        for i = 1 : numel(field_names) ,
          field_name = field_names{i} ;
          if isprop(obj, field_name) ,
            value = larva.(field_name) ;
            obj.set_property_value_(field_name, value) ;
          else
            warning('Unknown property %s', field_name) ;
          end
        end
      elseif isa(larva, 'DLBackEndClass') ,
        obj = larva ;
      else
        error('Unable to deal with a larva of class %s', class(larva)) ;
      end       
      if strcmp(obj.singularity_image_path_, '<invalid>') ,
        % This must come from an older .mat file, so we use the legacy values
        obj.singularity_image_path_ = DLBackEndClass.legacy_default_singularity_image_path ;
        obj.does_have_special_singularity_detection_image_path_ = true ;
        obj.singularity_detection_image_path_ = DLBackEndClass.legacy_default_singularity_image_path_for_detect ;
      end  
    end

    function jobid = parseJobID(backend_type, response)
      % Return the job id (as an old-style string) from the response to the system()
      % command spawning the job.
      switch backend_type
        case DLBackEnd.AWS,
          jobid = apt.parseJobIDAWS(response) ;
        case DLBackEnd.Bsub,
          jobid = apt.parseJobIDBsub(response) ;
        case DLBackEnd.Conda,
          jobid = apt.parseJobIDConda(response) ;
        case DLBackEnd.Docker,
          jobid = apt.parseJobIDDocker(response) ;
        otherwise
          error('Not implemented: %s',backend_type);
      end
    end    
  end  % methods (Static)

  methods
    function stopEc2InstanceIfNeeded_(obj)  % private by convention
      aws = obj.awsec2 ;
      % Sometimes .awsec2 is empty, even though that's not supposed to be possible
      % anymore.  Not clear to me how this happens.  -- ALT, 2024-10-09
      if isempty(aws) ,
        return
      end
      % DEBUGAWS: Stopping the AWS instance takes too long when debugging.      
      if aws.isInDebugMode ,
        return
      end
      tfsucc = aws.stopInstance();
      if ~tfsucc
        warningNoTrace('Failed to stop AWS EC2 instance %s.',aws.instanceID);
      end
    end  % function    

    function result = get.isInAwsDebugMode(obj)
      result = obj.awsec2.isInDebugMode ;
    end

    function set.isInAwsDebugMode(obj, value)
      obj.awsec2.isInDebugMode = value ;
    end    

    function killAndClearRegisteredJobs(obj, train_or_track)
      if strcmp(train_or_track, 'train') ,
        isTrain = true ;
      elseif strcmp(train_or_track, 'track') ,
        isTrain = false ;
      else
        error('DLBackEndClass:unknownJobType', 'The job type ''%s'' is not valid', train_or_track) ;
      end
      if isTrain 
        jobids = obj.training_jobids_ ;
      else
        jobids = obj.tracking_jobids_ ;
      end
      job_count = numel(jobids) ;
      for i = 1 : job_count ,
        jobid = jobids{i} ;
        if ~isempty(jobid) ,
          obj.ensureJobIsNotAlive(jobid) ;
        end
      end
      % Clear all registered jobs of the given type
      if isTrain
        obj.training_syscmds_ = cell(0,1) ;
        obj.training_cmdfiles_ = cell(0,1) ;
        obj.training_logcmds_ = cell(0,1) ;
        obj.training_jobids_ = cell(0,1) ;
      else
        obj.tracking_syscmds_ = cell(0,1) ;
        obj.tracking_cmdfiles_ = cell(0,1) ;
        obj.tracking_logcmds_ = cell(0,1) ;
        obj.tracking_jobids_ = cell(0,1) ;
      end
    end  % function

    function registerTrainingJob(backend, dmcjob, tracker, gpuids, do_just_generate_db)
      % Register a single training job with the backend, for later spawning via
      % spawnRegisteredJobs().

      % Get the root of the remote source tree
      remoteaptroot = backend.aptSourceDirRoot() ;
      
      ignore_local = (backend.type == DLBackEnd.Bsub) ;  % whether to pass the --ignore_local options to APTInterface.py
      basecmd = APTInterf.trainCodeGenBase(dmcjob,...
                                           'ignore_local',ignore_local,...
                                           'aptroot',remoteaptroot,...
                                           'do_just_generate_db',do_just_generate_db, ...
                                           'torchhome', backend.getTorchHome());
      args = determineArgumentsForSpawningJob(backend,tracker,gpuids,dmcjob,remoteaptroot,'train');
      syscmd = wrapCommandToBeSpawnedForBackend(backend,basecmd,args{:});
      cmdfile = DeepModelChainOnDisk.getCheckSingle(dmcjob.trainCmdfileLnx());
      logcmd = backend.generateLogCommand_('train', dmcjob) ;

      % Add all the commands to the registry
      backend.training_syscmds_{end+1,1} = syscmd ;
      backend.training_logcmds_{end+1,1} = logcmd ;
      backend.training_cmdfiles_{end+1,1} = cmdfile ;
      backend.training_jobids_{end+1,1} = [] ;  % indicates not-yet-spawned job
    end

    function registerTrackingJob(backend, totrackinfo, deeptracker, gpuids, track_type)
      % Register a single tracking job with the backend, for later spawning via
      % spawnRegisteredJobs().
      % track_type should be one of {'track', 'link', 'detect'}

      % Get the root of the remote source tree
      remoteaptroot = backend.aptSourceDirRoot() ;

      % totrackinfo has local paths, need to remotify them
      remotetotrackinfo = totrackinfo.copy() ;
      remotetotrackinfo.changePathsToRemoteFromLocal(backend.localDMCRootDir, backend) ;

      ignore_local = (backend.type == DLBackEnd.Bsub) ;  % whether to pass the --ignore_local options to APTInterface.py
      basecmd = APTInterf.trackCodeGenBase(totrackinfo,...
                                           'ignore_local',ignore_local,...
                                           'aptroot',remoteaptroot,...
                                           'track_type',track_type, ...
                                           'torchhome', backend.getTorchHome());
      args = determineArgumentsForSpawningJob(backend, deeptracker, gpuids, remotetotrackinfo, remoteaptroot, 'track') ;
      syscmd = wrapCommandToBeSpawnedForBackend(backend, basecmd, args{:}) ;
      cmdfile = DeepModelChainOnDisk.getCheckSingle(remotetotrackinfo.cmdfile) ;
      logcmd = backend.generateLogCommand_('track', remotetotrackinfo) ;
    
      % Add all the commands to the registry
      backend.tracking_syscmds_{end+1,1} = syscmd ;
      backend.tracking_logcmds_{end+1,1} = logcmd ;
      backend.tracking_cmdfiles_{end+1,1} = cmdfile ;
      backend.tracking_jobids_{end+1,1} = [] ;  % indicates not-yet-spawned job
    end

    function [didSpawnAllJobs, spawned_jobids] = spawnRegisteredJobs(obj, train_or_track, varargin)
      % Spawn all the training/tracking jobs that have been previously registered.
      % On entry, all jobs of the given type should be 
      [jobdesc, do_call_apt_interface_dot_py] = myparse( ...
        varargin, ...
        'jobdesc', 'job', ...
        'do_call_apt_interface_dot_py', true) ;

      % Sort out which registered jobs will be spawned.
      if strcmp(train_or_track, 'train') ,
        syscmds = obj.training_syscmds_ ;
        logcmds = obj.training_logcmds_ ;
        cmdfiles = obj.training_cmdfiles_ ;
      elseif strcmp(train_or_track, 'track') ,
        syscmds = obj.tracking_syscmds_ ;
        logcmds = obj.tracking_logcmds_ ;
        cmdfiles = obj.tracking_cmdfiles_ ;
      else
        error('DLBackEndClass:unknownJobType', 'The job type ''%s'' is not valid', train_or_track) ;
      end

      % Write the commands to files
      if ~isempty(cmdfiles),
        obj.writeCmdToFile(syscmds,cmdfiles,jobdesc);
      end

      % Actually spawn the jobs
      [didSpawnAllJobs, spawned_jobids] = DLBackEndClass.spawnJobs(syscmds, logcmds, obj.type, jobdesc, do_call_apt_interface_dot_py) ;

      % If all went well, record the spawned jobids.  If not, kill any straggler
      % jobs.
      if didSpawnAllJobs ,
        if strcmp(train_or_track, 'train') ,
          obj.training_jobids_ = spawned_jobids ;  % Keep these around internally so we can kill the jobs on delete()
        elseif strcmp(train_or_track, 'track') ,
          obj.tracking_jobids_ = spawned_jobids ;  % Keep these around internally so we can kill the jobs on delete()
        else
          error('DLBackEndClass:unknownJobType', 'The job type ''%s'' is not valid', train_or_track) ;
        end
      else
        % If not all jobs were successfully spawned, kill all the jobs that *were* spawned
        job_count = numel(spawned_jobids) ;
        for i = 1 : job_count ,
          jobid = spawned_jobids{i} ;
          if ~isempty(jobid) ,
            obj.ensureJobIsNotAlive(jobid) ;
          end
        end
      end  % if        
    end  % function
  end  % methods

  methods (Static)
    function [didSpawnAllJobs, jobidFromJobIndex] = spawnJobs(syscmds, logcmds, backend_type, jobdesc, do_call_apt_interface_dot_py)
      % Spawn the jobs specified in syscmds.  On return, didSpawnAllJobs indicates
      % whether all went well.  *Regardless of the value of didSpawnAllJobs,*
      % jobidFromJobIndex contains the jobids of all the jobs that were successfully
      % spawned.  
      jobCount = numel(syscmds) ;
      jobidFromJobIndex = cell(0,1) ;  % We only add jobids to this once they have successfully been spawned.
      for jobIndex = 1:jobCount ,
        syscmd = syscmds{jobIndex} ;
        fprintf('%s\n',syscmd);
        if do_call_apt_interface_dot_py ,
          [rc,stdouterr] = apt.syscmd(syscmd, 'failbehavior', 'silent');
          didSpawn = (rc == 0) ;
          if didSpawn ,
            jobid = DLBackEndClass.parseJobID(backend_type, stdouterr);
          else
            didSpawnAllJobs = false ;
            return
          end
        else
          % Pretend it's a failure, for expediency.
          didSpawnAllJobs = false ;
          return
        end

        % If get here, this job spawn succeeded
        jobidFromJobIndex{jobIndex,1} = jobid ;  % Add to end, keeping it a col vector
        assert(ischar(jobid)) ;
        fprintf('%s %d spawned, ID = %s\n\n',jobdesc,jobIndex,jobid);

        % Now give the command to create the log file.
        % (This only does anything interesting for a docker backend.)
        if numel(logcmds) >= jobIndex ,
          logcmd = logcmds{jobIndex} ;
          if ~isempty(logcmd) ,
            fprintf('%s\n',logcmd);
            [rc2,stdouterr2] = apt.syscmd(logcmd, 'failbehavior', 'silent');
            didLogCommandSucceed = (rc2==0) ;
            if ~didLogCommandSucceed ,
              % Throw a warning here, but proceed anyway.
              % I have never had occasion to look at one of these docker log files,
              % But presumably they're useful sometimes.  -- ALT, 2025-01-23
              warning('Failed to spawn logging for %s %d: %s.',jobdesc,jobIndex,stdouterr2);
            end
          end
        end  % if
      end  % for      
      didSpawnAllJobs = true ;  % if get here, all is well
    end  % function
  end  % methods (Static)

  methods
    function waitForRegisteredJobsToExit(obj, train_or_track)
      % Wait for registered training/tracking jobs to exit.
      if strcmp(train_or_track, 'train') ,
        jobids = obj.training_jobids_ ;
      elseif strcmp(train_or_track, 'track') ,
        jobids = obj.tracking_jobids_ ;
      else
        error('DLBackEndClass:unknownJobType', 'The job type ''%s'' is not valid', train_or_track) ;
      end
      job_count = numel(jobids) ;
      for i = 1 : job_count ,
        jobid = jobids{i} ;
        if ~isempty(jobid) ,
          obj.waitForJobToExit(jobid) ;
        end
      end
    end  % function
    
    function waitForJobToExit(obj, jobid)
      % Wait for the job with job id jobid to exit.
      % jobid is assumed to be a single job id, represented as an old-style string.      
      if isempty(jobid) ,
        error('Job id is empty') ;
      end
      while obj.isJobAlive(jobid) ,
        pauseTight(0.25) ;
      end
    end  % function

    function ensureJobIsNotAlive(obj, jobid)
      % Kill the job with job id jobid.
      % jobid is assumed to be a single job id, represented as an old-style string.      
      if isempty(jobid) ,
        error('Job id is empty') ;
      end
      if obj.isJobAlive(jobid) ,
        obj.killJob(jobid) ;
      end
    end  % function

    function result = isAliveFromRegisteredJobIndex(obj, train_or_track)
      if strcmp(train_or_track, 'train') ,
        jobids = obj.training_jobids_ ;
      elseif strcmp(train_or_track, 'track') ,
        jobids = obj.tracking_jobids_ ;
      else
        error('DLBackEndClass:unknownJobType', 'The job type ''%s'' is not valid', train_or_track) ;
      end
      result = cellfun(@(jobid)(obj.isJobAlive(jobid)), jobids) ;  % boolean array
    end  % function
    
    function tf = isJobAlive(obj, jobid)
      % Returns true if there is a running job with ID jobid.
      % jobid is assumed to be a single job id, represented as an old-style string.
      if isempty(jobid) ,
        error('Job id is empty');
      end
      if obj.type == DLBackEnd.AWS || obj.type == DLBackEnd.Docker ,
        tf = obj.isJobAliveDockerOrAWS_(jobid) ;
      elseif obj.type == DLBackEnd.Bsub ,
        tf = obj.isJobAliveBsub_(jobid) ;
      elseif obj.type == DLBackEnd.Conda ,
        tf = obj.isJobAliveConda_(jobid) ;
      else
        error('Unknown DLBackEnd value') ;
      end      
    end  % function

    function tf = isJobAliveDockerOrAWS_(obj, jobid)
      % Returns true if there is a running job with ID jobid.
      % jobid is assumed to be a single job id, represented as an old-style string.      
      jobidshort = jobid(1:8);
      cmd = sprintf('%s ps -q -f "id=%s"',apt.dockercmd(),jobidshort);      
      [st,res] = obj.runBatchCommandOutsideContainer(cmd);
        % It uses the docker executable, but it still runs outside the docker
        % container.
      if st==0
        tf = ~isempty(regexp(res,jobidshort,'once')) ;
      else
        error('Error occurred when checking if %s job %s was running: %s', char(obj.type), jobid, res) ;
      end
    end  % function   
    
    function tf = isJobAliveBsub_(obj, jobid)  %#ok<INUSD> 
      % Returns true if there is a running job with ID jobid.
      % jobid is assumed to be a single job id, represented as an old-style string.
      runStatuses = {'PEND','RUN','PROV','WAIT'};
      pollcmd0 = sprintf('bjobs -o stat -noheader %s',jobid);
      pollcmd = wrapCommandSSH(pollcmd0,'host',DLBackEndClass.jrchost);
      %[st,res] = system(pollcmd);
      [st,res] = apt.syscmd(pollcmd, 'failbehavior', 'silent', 'verbose', false) ;
      if st==0
        s = sprintf('(%s)|',runStatuses{:});
        s = s(1:end-1);
        tf = ~isempty(regexp(res,s,'once'));
      else
        error('Error occurred when checking if bsub job %s was running: %s', jobid, res) ;
      end
    end  % function

    function tf = isJobAliveConda_(obj, jobid)  %#ok<INUSD>
      % Returns true if there is a running conda job with ID jobid.
      % jobid is assumed to be a single job id, represented as an old-style string.      
      command_line = sprintf('/usr/bin/pgrep --pgroup %s', jobid) ;  % For conda backend, the jobid is a PGID
      [return_code, stdouterr] = system(command_line) ;  %#ok<ASGLU>  % conda is Linux-only, so can just use system()
      % pgrep exits with return_code == 1 if there is no such PGID.  Not great for
      % detecting when something *else* has gone wrong, but whaddayagonnado?
      % We capture stdouterr to prevent it getting spit out to the Matlab console.
      % We use a variable name instead of ~ in case we need to debug in here at some
      % point.
      tf = (return_code == 0) ;
    end  % function

    function killJob(obj, jobid)
      % Kill the job with job id jobid.
      % jobid is assumed to be a single job id, represented as an old-style string.      
      if isempty(jobid) ,
        error('Job id is empty');
      end
      if obj.type == DLBackEnd.AWS || obj.type == DLBackEnd.Docker ,
        obj.killJobDockerOrAWS_(jobid) ;
      elseif obj.type == DLBackEnd.Bsub ,
        obj.killJobBsub_(jobid) ;
      elseif obj.type == DLBackEnd.Conda ,
        obj.killJobConda_(jobid) ;
      else
        error('Unknown DLBackEnd value') ;
      end      
    end  % function

    function killJobBsub_(obj, jobid)  %#ok<INUSD> 
      % Kill the bsub job with job id jobid.
      % jobid is assumed to be a single job id, represented as an old-style string.      
      bkillcmd0 = sprintf('bkill %s',jobid);
      bkillcmd = wrapCommandSSH(bkillcmd0,'host',DLBackEndClass.jrchost);
      [st,res] = apt.syscmd(bkillcmd, 'failbehavior', 'silent', 'verbose', false) ;
      if st~=0 ,
        error('Error occurred when trying to kill bsub job %s: %s', jobid, res) ;
      end
    end  % function

    function killJobConda_(obj, jobid)  %#ok<INUSD> 
      pgid = jobid ;  % conda backend uses PGID as the job id
      command_line = sprintf('kill -- -%s', pgid) ;  % kill all processes in the process group
      system_with_error_handling(command_line) ;  % conda is Linux-only, so can just use system()
    end  % function

    function killJobDockerOrAWS_(obj, jobid)
      % Kill the docker job with job id jobid.
      % jobid is assumed to be a single job id, represented as an old-style string.      
      % Errors if no such job exists.
      cmd = sprintf('%s kill %s', apt.dockercmd(), jobid);        
      [st,res] = obj.runBatchCommandOutsideContainer(cmd) ;
        % It uses the docker executable, but it still runs outside the docker
        % container.
      if st~=0 ,
        error('Error occurred when trying to kill %s job %s: %s', char(obj.type), jobid, res) ;
      end
    end  % function

    function [isAllWell, message] = downloadTrackingFilesIfNecessary(obj, res, localCacheRoot, movfiles)
      if obj.type == DLBackEnd.AWS ,
        [isAllWell, message] = obj.awsec2.downloadTrackingFilesIfNecessary(res, localCacheRoot, movfiles) ;
      elseif obj.type == DLBackEnd.Bsub ,
        % Hack: For now, just wait a bit, to let (hopefully) NFS sync up
        pause(10) ;
        isAllWell = true ;
        message = '' ;
      elseif obj.type == DLBackEnd.Conda ,
        isAllWell = true ;
        message = '' ;
      elseif obj.type == DLBackEnd.Docker ,
        if ~isempty(obj.dockerremotehost) ,
          % This path is for when the docker backend is running on a remote host.
          % Hack: For now, just wait a bit, to let (hopefully) NFS sync up.
          pause(10) ;
        end          
        isAllWell = true ;
        message = '' ;
      else
        error('Internal error: Unknown DLBackEndClass type') ;
      end
    end  % function    

    % function setAwsPemFileAndKeyName(obj, pemFile, keyName)
    %   ec2 = obj.awsec2 ;
    %   ec2.pem = pemFile ;
    %   ec2.keyName = keyName ;
    % end
    
    % function setAWSInstanceIDAndType(obj, instanceID, instanceType)
    %   ec2 = obj.awsec2 ;
    %   ec2.setInstanceIDAndType(instanceID, instanceType) ;
    % end
    
    function [tfsucc,res] = batchPoll(obj, fspollargs)
      % fspollargs: [n] cellstr eg {'exists' '/my/file' 'existsNE' '/my/file2'}
      %
      % res: [n] cellstr of fspoll responses

      if obj.type == DLBackEnd.AWS ,
        [tfsucc,res] = obj.awsec2.batchPoll(fspollargs) ;
      else
        error('Not implemented') ;        
        %fspoll_script_path = linux_fullfile(APT.Root, 'matlab/misc/fspoll.py') ;
      end
    end  % function
    
    function result = fileExists(obj, file_name)
      % Returns true iff the named file exists.
      % Should be consolidated with exist(), probably.  Note, though, that probably
      % need to be careful about checking for the file inside/outside the container.
      if obj.type == DLBackEnd.AWS ,
        result = obj.awsec2.fileExists(file_name) ;
      else
        result = logical(exist(file_name,'file')) ;
      end
    end  % function

    function result = fileExistsAndIsNonempty(obj, file_name)
      % Returns true iff the named file exists and is not zero-length.
      if obj.type == DLBackEnd.AWS ,
        result = obj.awsec2.fileExistsAndIsNonempty(file_name) ;
      else
        result = localFileExistsAndIsNonempty(file_name) ;
      end
    end  % function

    function result = fileExistsAndIsGivenSize(obj, file_name, sz)
      % Returns true iff the named file exists and is the given size (in bytes).
      if obj.type == DLBackEnd.AWS ,
        result = obj.awsec2.fileExistsAndIsGivenSize(file_name, sz) ;
      else
        result = localFileExistsAndIsGivenSize(file_name, sz) ;
      end
    end  % function

    function result = fileContents(obj, file_name)
      % Return the contents of the named file, as an old-style string.
      % The behavior of this function when the file does not exist is kinda weird.
      % It is the way it is b/c it's designed for giving something helpful to
      % display in the monitor window.
      if obj.type == DLBackEnd.AWS ,
        result = obj.awsec2.fileContents(file_name) ;
      else
        if exist(file_name,'file') ,
          lines = readtxtfile(file_name);
          result = sprintf('%s\n',lines{:});
        else
          result = '<file does not exist>';
        end
      end
    end  % function
    
    function lsdir(obj, dir)
      % List the contents of directory dir.  Contents just go to stdout, nothing is
      % returned.
      if obj.type == DLBackEnd.AWS ,
        obj.awsec2.lsdir(dir);
      else
        if ispc()
          lscmd = 'dir';
        else
          lscmd = 'ls -al';
        end
        cmd = sprintf('%s "%s"',lscmd,dir);
        system(cmd);
      end
    end  % function

    function result = fileModTime(obj, file_name)
      % Return the file-modification time (mtime) of the given file.  For an AWS
      % backend, this is the file modification time in seconds since epoch.  For
      % other backends, it's a Matlab datenum of the mtime.  So these should not be
      % compared across backend types.
      if obj.type == DLBackEnd.AWS ,
        result = obj.awsec2.remoteFileModTime(file_name) ;
      else
        dir_struct = dir(file_name) ;
        result = dir_struct.datenum ;
      end
    end  % function

    function suitcase = packParfevalSuitcase(obj)
      % Use before calling parfeval, to restore Transient properties that we want to
      % survive the parfeval boundary.
      if obj.type == DLBackEnd.AWS ,
        suitcase = obj.awsec2.packParfevalSuitcase() ;
      else
        suitcase = [] ;
      end
    end  % function
    
    function restoreAfterParfeval(obj, suitcase)
      % Should be called in background tasks run via parfeval, to restore fields that
      % should not be restored from persistence, but we want to survive the parfeval
      % boundary.
      if obj.type == DLBackEnd.AWS ,
        obj.awsec2.restoreAfterParfeval(suitcase) ;
      else
        % do nothing
      end
    end  % function
    
    % function result = scpDownloadOrVerify(obj, srcAbs, dstAbs, varargin)
    %   if obj.type == DLBackEnd.AWS ,
    %     result = obj.awsec2.scpDownloadOrVerify(srcAbs, dstAbs, varargin) ;
    %   else
    %     result = true ;
    %   end        
    % end  % function
    % 
    % function result = scpDownloadOrVerifyEnsureDir(obj, srcAbs, dstAbs, varargin)
    %   if obj.type == DLBackEnd.AWS ,
    %     result = obj.awsec2.scpDownloadOrVerifyEnsureDir(srcAbs, dstAbs, varargin) ;
    %   else
    %     result = true ;
    %   end        
    % end  % function

    function nframes = readTrkFileStatus(obj, filename, partFileIsTextStatus, logger)
      % Read the number of frames remaining according to the remote file at location
      % filename.  If partFileIsTextStatus is true, this file is assumed to be a
      % text file.  Otherwise, it is assumed to be a .mat file.
      if ~exist('partFileIsTextStatus', 'var') || isempty(partFileIsTextStatus) ,
        partFileIsTextStatus = false ;
      end
      if ~exist('logger', 'var') || isempty(logger) ,
        logger = FileLogger(1, 'DLBackEndClass::readTrkFileStatus()') ;
      end

      if obj.type == DLBackEnd.AWS ,
        %logger.log('partFileIsTextStatus: %d', double(partFileIsTextStatus)) ;
        nframes = 0;
        if ~obj.fileExists(filename) ,
          return
        end
        if partFileIsTextStatus,
          str = obj.fileContents(filename) ;
          nframes = TrkFile.getNFramesTrackedString(str) ;
        else
          local_filename = strcat(tempname(), '.mat') ;  % Has to have an extension or matfile() will add '.mat' to the filename
          %logger.log('BgTrackWorkerObjAWS::readTrkFileStatus(): About to call obj.awsec2.scpDownloadOrVerify()...\n') ;
          did_succeed = obj.awsec2.scpDownloadOrVerify(filename, local_filename) ;
          %logger.log('BgTrackWorkerObjAWS::readTrkFileStatus(): Returned from call to obj.awsec2.scpDownloadOrVerify().\n') ;
          if did_succeed ,
            %logger.log('Successfully downloaded remote tracking file %s\n', filename) ;
            try
              nframes = TrkFile.getNFramesTrackedMatFile(local_filename) ;
            catch me
              logger.log('Could not read tracking progress from remote file %s: %s\n', filename, me.message) ;
            end
            %logger.log('Read that nframes = %d\n', nframes) ;
          else
            logger.log('Could not download tracking progress from remote file %s\n', filename) ;
          end
        end
      else
        % If non-AWS backend
        nframes = 0;
        if ~exist(filename,'file'),
          return;
        end
        if partFileIsTextStatus ,
          s = obj.fileContents(filename) ;
          nframes = TrkFile.getNFramesTrackedPartFile(s) ;
        else
          try
            nframes = TrkFile.getNFramesTrackedMatFile(filename);
          catch
            fprintf('Could not read tracking progress from %s\n',filename);
          end
        end        
      end
    end  % function
    
    function cmdfull = wrapCommandSSHAWS(obj, cmdremote, varargin)
      cmdfull = obj.awsec2.wrapCommandSSH(cmdremote, varargin{:}) ;
    end

    function maxiter = getMostRecentModel(obj, dmc)  % constant method
      if obj.type == DLBackEnd.AWS ,
        maxiter = obj.awsec2.getMostRecentModel(dmc) ;
      else
        maxiter = dmc.getMostRecentModelLocal() ;
      end
    end  % function
    
    function mirrorDMCToBackend(obj, dmc, mode)
      % mode should be 'tracking' or 'training'.
      if ~exist('mode', 'var') || isempty(mode) ,
        mode = 'tracking' ;
      end
      assert(isa(dmc, 'DeepModelChainOnDisk')) ;      
      if obj.type == DLBackEnd.AWS ,
         obj.awsec2.mirrorDMCToBackend(dmc, mode) ;
      end
    end

    function mirrorDMCFromBackend(obj, dmc)
      % If the model chain is remote, download it
      assert(isa(dmc, 'DeepModelChainOnDisk')) ;      
      if obj.type == DLBackEnd.AWS ,
         obj.awsec2.mirrorDMCFromBackend(dmc) ;
      end
    end  % function

    function result = get.isDMCRemote(obj)
      result = (obj.type == DLBackEnd.AWS) && obj.awsec2.isDMCRemote ;
    end  % function

    function result = get.isDMCLocal(obj)
      result = ~obj.isDMCRemote ;
    end  % function

    function prepareFilesForTracking(backend, toTrackInfo)
      backend.ensureFoldersNeededForTrackingExist_(toTrackInfo);
      backend.ensureFilesDoNotExist_({toTrackInfo.getErrfile()}, 'error file');
      backend.ensureFilesDoNotExist_(toTrackInfo.getPartTrkFiles(), 'partial tracking result');
      backend.ensureFilesDoNotExist_({toTrackInfo.getKillfile()}, 'kill files');
    end  % function

    function ensureFoldersNeededForTrackingExist_(obj, toTrackInfo)
      dirlocs = toTrackInfo.trkoutdir ;
      desc = 'trk cache dir' ;
      for i = 1:numel(dirlocs),
        dirloc = dirlocs{i} ;
        if ~obj.fileExists(dirloc) ,
          [succ,msg] = obj.mkdir(dirloc);
          if ~succ
            error('Failed to create %s %s: %s',desc,dirloc,msg);
          else
            fprintf('Created %s: %s\n',desc,dirloc);
          end
        end
      end
    end  % function

    function ensureFilesDoNotExist_(obj, filelocs, desc)
      for i = 1:numel(filelocs),
        fileloc = filelocs{i} ;
        if obj.fileExists(fileloc),
          fprintf('Deleting %s %s',desc,fileloc);
          obj.deleteFile(fileloc);
        end
        if obj.fileExists(fileloc),
          error('Failed to delete %s: file still exists',fileloc);
        end
      end
    end  % function

    function result = get.localDMCRootDir(obj) 
      result = obj.awsec2.localDMCRootDir ;
    end  % function

    function set.localDMCRootDir(obj, value) 
      obj.awsec2.localDMCRootDir = value ;
    end  % function

    function result = get.remoteDMCRootDir(obj)  %#ok<MANU>
      result = AWSec2.remoteDLCacheDir ;
    end  % function

    function result = get.awsInstanceID(obj)
      result = obj.awsec2.instanceID ;
    end  % function

    function result = get.awsKeyName(obj)
      result = obj.awsec2.keyName ;
    end  % function

    function result = get.awsPEM(obj)
      result = obj.awsec2.pem ;
    end  % function

    function result = get.awsInstanceType(obj)
      result = obj.awsec2.instanceType ;
    end  % function

    function set.awsInstanceID(obj, value)
      obj.awsec2.instanceID = value ;
    end  % function

    function set.awsKeyName(obj, value)
      obj.awsec2.keyName = value ;
    end  % function

    function set.awsPEM(obj, value)
      obj.awsec2.pem = value ;
    end  % function

    function set.awsInstanceType(obj, value)
      obj.awsec2.instanceType = value ;
    end  % function
    
    function result = getTorchHome(obj)
      if obj.type == DLBackEnd.AWS ,
        result = obj.awsec2.getTorchHome() ;
      else
        result = fullfile(APT.getdotaptdirpath(), 'torch') ;
      end
    end  % function   
    
    function statusStringFromJobIndex = queryAllJobsStatus(obj, train_or_track)
      % Returns a cell array of status strings, one for each spawned job.
      % Each line is of the form 'Job 12345 is alive' or 'Job 12345 is dead'.
      if strcmp(train_or_track, 'train') ,
        jobIDFromJobIndex = obj.training_jobids_ ;
      elseif strcmp(train_or_track, 'track') ,
        jobIDFromJobIndex = obj.tracking_jobids_ ;
      else
        error('DLBackEndClass:unknownJobType', 'The job type ''%s'' is not valid', train_or_track) ;
      end
      isAliveFromJobIndex = obj.isAliveFromRegisteredJobIndex(train_or_track) ;
      livenessStringFromJobIndex = arrayfun(@(isAlive)(fif(isAlive, 'alive', 'dead')), isAliveFromJobIndex, 'UniformOutput', false) ;
      statusStringFromJobIndex = cellfun(@(jobID, livenessString)(sprintf('Job %s is %s', jobID, livenessString)), ...
                                         jobIDFromJobIndex, livenessStringFromJobIndex, ...
                                         'UniformOutput', false) ;
    end  % function    
    
    function logcmd = generateLogCommand_(obj, train_or_track, dmcjob_or_totrackinfojob)  % constant method
      if strcmp(train_or_track, 'train') ,
        dmcjob = dmcjob_or_totrackinfojob ;
        if obj.type == DLBackEnd.Docker ,
          containerName = DeepModelChainOnDisk.getCheckSingle(dmcjob.trainContainerName) ;
          logfile = DeepModelChainOnDisk.getCheckSingle(dmcjob.trainLogLnx) ;
          logcmd = obj.generateLogCommandForDockerBackend_(containerName, logfile) ;
        else
          logcmd = '' ;
        end
      elseif strcmp(train_or_track, 'track') ,
        totrackinfojob = dmcjob_or_totrackinfojob ;
        if obj.type == DLBackEnd.Docker ,
          containerName = totrackinfojob.containerName ;
          logfile = totrackinfojob.logfile ;
          logcmd = obj.generateLogCommandForDockerBackend_(containerName, logfile) ;
        else
          logcmd = '' ;
        end
      else
        error('train_or_track had illegal value ''%s''', train_or_track) ;
      end
    end  % function

    function cmd = generateLogCommandForDockerBackend_(backend, containerName, native_log_file_name)  % constant method
      assert(backend.type == DLBackEnd.Docker);
      dockercmd = apt.dockercmd();
      log_file_name = linux_path(native_log_file_name) ;
      cmd = ...
        sprintf('%s logs -f %s &> %s', ... 
                dockercmd, ...
                containerName, ...
                escape_string_for_bash(log_file_name)) ;
      is_docker_remote = ~isempty(backend.dockerremotehost) ;
      if is_docker_remote
        cmd = wrapCommandSSH(cmd,'host',backend.dockerremotehost);
      end
      cmd = sprintf('%s &', cmd);
    end  % function

    function result = detailedStatusStringFromRegisteredJobIndex(obj, train_or_track)
      if strcmp(train_or_track, 'train') ,
        jobids = obj.training_jobids_ ;
      elseif strcmp(train_or_track, 'track') ,
        jobids = obj.tracking_jobids_ ;
      else
        error('DLBackEndClass:unknownJobType', 'The job type ''%s'' is not valid', train_or_track) ;
      end
      result = cellfun(@(jobid)(obj.detailedStatusString(jobid)), jobids, 'UniformOutput', false) ;  % cell array of old-style strings
    end  % function
    
    function result = detailedStatusString(obj, jobid)
      % Returns a detailed status string for the job with ID jobid.
      % jobid is assumed to be a single job id, represented as an old-style string.
      if isempty(jobid) ,
        error('Job id is empty');
      end
      if obj.type == DLBackEnd.AWS || obj.type == DLBackEnd.Docker ,
        result = obj.detailedStatusStringDockerOrAWS_(jobid) ;
      elseif obj.type == DLBackEnd.Bsub ,
        result = obj.detailedStatusStringBsub_(jobid) ;
      elseif obj.type == DLBackEnd.Conda ,
        result = obj.detailedStatusStringConda_(jobid) ;
      else
        error('Unknown DLBackEnd value') ;
      end      
    end  % function

    function result = detailedStatusStringDockerOrAWS_(obj, jobid)
      % Returns true if there is a running job with ID jobid.
      % jobid is assumed to be a single job id, represented as an old-style string.      
      jobidshort = jobid(1:8) ;
      cmd = sprintf('%s ps --filter "id=%s"', apt.dockercmd(), jobidshort) ;      
      [rc, stdouterr] = obj.runBatchCommandOutsideContainer(cmd) ;
        % It uses the docker executable, but it still runs outside the docker
        % container.
      if rc==0
        result = stdouterr ;
      else
        result = sprintf('Error occurred when checking if docker job %s was running: %s', jobid, stdouterr) ;
      end
    end  % function   
    
    function result = detailedStatusStringBsub_(obj, jobid)
      % Returns true if there is a running job with ID jobid.
      % jobid is assumed to be a single job id, represented as an old-style string.
      cmd0 = sprintf('bjobs %s', jobid) ;
      cmd1 = wrapCommandSSH(cmd0, 'host', DLBackEndClass.jrchost) ;
        % For the bsub backend, obj.runBatchCommandOutsideContainer() still runs
        % things locally, since that's what you want for e.g. commands that check on
        % file status.
      [rc, stdouterr] = obj.runBatchCommandOutsideContainer(cmd1) ;
      if rc==0 ,
        result = stdouterr ;
      else
        result = sprintf('Error occurred when checking status of bsub job %s: %s', jobid, stdouterr) ;
      end
    end  % function

    function result = detailedStatusStringConda_(obj, jobid)  %#ok<INUSD>
      % Returns true if there is a running conda job with ID jobid.
      % jobid is assumed to be a single job id, represented as an old-style string.      
      command_line = sprintf('/usr/bin/pgrep --pgroup %s', jobid) ;  % For conda backend, the jobid is a PGID
      [return_code, stdouterr] = system(command_line) ;  %#ok<ASGLU>  % conda is Linux-only, so can just use system()
      % pgrep exits with return_code == 1 if there is no such PGID.  Not great for
      % detecting when something *else* has gone wrong, but whaddayagonnado?
      % We capture stdouterr to prevent it getting spit out to the Matlab console.
      % We use a variable name instead of ~ in case we need to debug in here at some
      % point.
      if rc==0 ,
        result = stdouterr ;
      else
        result = sprintf('Error occurred when checking status of conda job %s: %s', jobid, stdouterr) ;
      end
    end  % function


  end  % methods
end  % classdef
