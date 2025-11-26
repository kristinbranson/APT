classdef DLBackEndClass < handle
  % APT Backends specify a physical machine/server where GPU code is run.
  % This class is intended to abstract away details particular to one backend or
  % another, so that calling code doesn't need to worry about such grubby
  % details.
  %
  % Note that filesystem paths stored in a DLBackEndClass object should be
  % apt.MetaPath values with wsl locale (see the docs for that class).  All
  % apt.MetaPath objects have locale of either a) native, b) WSL, or c) remote.
  % A native metapath points to a file in the native filesystem of the frontend
  % (either Linux or Windows).  E.g. /home/joeuser/data or
  % C:\Users\joeuser\Documents\data.  A WSL metapath represents a valid path inside
  % the local Linux environment.  On Windows, that's WSL2. On Linux, that's just
  % Linux, so on Linux the WSL path is just the native path. So on Windows the
  % native path 'C:\Users\joeuser\Documents\data' would become the WSL path
  % '/mnt/c/Users/joeuser/Documents/data'.  A remote path is a path on the
  % remote backend filesystem, for instance AWS.  For the docker backend the WSL
  % path and the remote path will be the same.  Ditto the conda and bsub
  % backends.  For the AWS backend the remote path will typically start with
  % '/home/ubuntu'.
  %
  % For the Labeler and the DeepTracker, all paths stored in them should be
  % native paths (sometimes as char arrays, although hopefully that usage will
  % decrease going forward), and all paths they pass to DLBackendClass methods
  % should be native paths, however represented.  All filesystem paths stored in
  % the DLBackEndClass proper should be apt.MetaPath objects with WSL locale.
  % All paths stored in the AWSec2 object will be apt.MetaPath objects with
  % remote locale, except where noted.  All paths passed from the DLBackendClass
  % object to the AWSec2 object should be WSL apt.MetaPath objects with WSL
  % locale.
  %
  % Also note that when synthesizing command lines, these should normally be
  % represented as apt.ShellCommand objects with WSL locale.  The AWSec2 object
  % will convert these to remote apt.ShellCommand objects when needed.  These
  % objects are translated into actual command line strings at the last possible
  % moment, in the apt.CommandShell.run() method.
  %
  % And another thing: Note that objects of this class (DLBackEndClass) have to
  % be copied (in a certain sense) to a parallel process to do polling of
  % training/tracking.  So it is by design that this class does not have a
  % .parent_ instance variable, b/c once you have one of those you generally
  % have an object that is not copyable, at least not in an obvious and
  % straightforward way.

  properties (Constant)
    minFreeMem = 9000  % in MiB
    defaultDockerImgTag = 'apt-20250626-tf215-pytorch21-hopper'
    defaultDockerImgRoot = 'bransonlabapt/apt_docker'
 
    jrchost = 'login1.int.janelia.org'
    % jrcprodrepo = '/groups/branson/bransonlab/apt/repo/prod'
    default_jrcgpuqueue = 'gpu_a100'
    default_jrcnslots_train = 4
    default_jrcnslots_track = 4

    default_conda_env = 'apt-20250626-tf215-pytorch21-hopper'
    DEFAULT_SINGULARITY_IMAGE_PATH = apt.MetaPath('/groups/branson/bransonlab/apt/sif/apt-20250626-tf215-pytorch21-hopper.sif', 'wsl', 'universal')
      % Since this path's filerole is universal, the locale doesn't really
      % matter much.  We use wsl to maintain consistency with the internal storage invariant.
  end

  properties
    type  % scalar DLBackEnd

    % Used only for type==Bsub
    % deepnetrunlocal = true
    %   % scalar logical. if true, bsub backend runs code in APT.Root/deepnet.
    %   % This path must be visible in the backend or else.
    %   % Applies only to bsub. Name should be eg 'bsubdeepnetrunlocal'
    % bsubaptroot = []  % root of APT repo for bsub backend running     
    jrcsimplebindpaths = true  % whether to bind '/groups', '/nrs' for the Bsub/JRC backend
        
    % Used only for type==AWS
    awsec2  % a scalar AWSec2 object (present whether we need it or not)
    
    % Used only for type==Docker  
    %dockerapiver = DLBackEndClass.default_docker_api_version  % docker codegen will occur against this docker api ver
    % dockerimgroot = DLBackEndClass.defaultDockerImgRoot
    %   % We have an instance prop for this to support running on older/custom
    %   % docker images.
    % dockerimgtag = DLBackEndClass.defaultDockerImgTag
    dockerremotehost = ''
      % The docker backend can run the docker container on a remote host.
      % dockerremotehost will contain the DNS name of the remote host in this case.
      % But even in this case, as with local docker, we assume that the docker
      % container and the local host have the same filesystem paths to all the
      % training/tracking files we will access.  (Like if e.g. they're both managed
      % Linux boxes on the Janelia network.)

    gpuids = []  % for now used by docker/conda
    
    jrcAdditionalBsubArgs = ''  % Additional arguments to be passed to JRC bsub command, e.g. '-P scicompsoft'    
    jrcgpuqueue 
    jrcnslots 
    jrcnslotstrack

    % condaEnv = DLBackEndClass.default_conda_env   % used only for Conda

    %singularity_image_path_ = ''
    %does_have_special_singularity_detection_image_path_ = '<invalid>'
    %singularity_detection_image_path_ = '<invalid>'
  end

  properties  % these are SetAccess=private by gentleperson's agreement
    % These keep track of whether we use the default image specs, or the custom
    % ones.
    didOverrideDefaultDockerImgSpec_ = false
    customDockerImgRoot_ = ''
    customDockerImgTag_ = ''
    didOverrideDefaultCondaEnv_ = false    
    customCondaEnv_ = ''
    didOverrideDefaultSingularityImagePath_ = false
    customSingularityImagePath_ = DLBackEndClass.DEFAULT_SINGULARITY_IMAGE_PATH  % Want this to have the right type (apt.MetaPath)
  end

  properties (Transient)
    % The job registry.  These are protected in spirit.
    % These are jobs that can be spawned with a subsequent call to
    % spawnRegisteredJobs().
    training_syscmds_ = cell(0,1)
    training_cmdfiles_ = cell(0,1)
    % training_logcmds_ = cell(0,1)
    tracking_syscmds_ = cell(0,1)
    tracking_cmdfiles_ = cell(0,1)
    % tracking_logcmds_ = cell(0,1)

    % The job IDs.  These are protected, in spirit.
    % Each job id is represented as an old-style *string*.  What exactly they mean
    % depends on the backend.  For conda backend, the job id is the PGID of the
    % process group of the Python APT_interface.py invocation.  For Docker and
    % AWS, it's the Docker process ID.  For bsub, it's the LSF job number.
    training_jobids_ = cell(0,1)
    tracking_jobids_ = cell(0,1)

    % Hold the text shown in the backend test window.
    % It is a cell array of strings.
    testText_ = cell(0,1)

    % This is used to keep track of whether we need to release/delete resources on
    % delete()
    doesOwnResources_ = true  % is obj a copy, or the original
  end

  properties (Dependent)
    dockerimgfull % full docker img spec (with tag if specified)
    singularity_image_path
    % singularity_detection_image_path
    isInAwsDebugMode
    isProjectCacheRemote
    isProjectCacheLocal
    wslProjectCachePath  % a MetaPath with wsl locale
    nativeProjectCachePath  % a MetaPath with native locale
    remoteDMCRootDir
    awsInstanceID
    awsKeyName  % key(pair) name used to authenticate to AWS EC2, e.g. 'alt_taylora-ws4'
    awsPEM  % absolute *WSL* path of .pem file that holds an RSA private key used to ssh into the AWS EC2 instance
    awsInstanceType
    condaEnv
    dockerimgroot
    dockerimgtag    
  end
  
  methods
    function obj = DLBackEndClass(ty)
      if ~exist('ty', 'var') || isempty(ty) ,
        ty = DLBackEnd.Bsub ;
      end
      obj.type = ty ;

      % set jrc backend fields to valid values
      if isempty(obj.jrcgpuqueue),
        obj.jrcgpuqueue = DLBackEndClass.default_jrcgpuqueue ;
      end
      if isempty(obj.jrcnslots) ,
        obj.jrcnslots = DLBackEndClass.default_jrcnslots_train ;
      end
      if isempty(obj.jrcnslotstrack) ,
        obj.jrcnslotstrack = DLBackEndClass.default_jrcnslots_track ;
      end      

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

    function set.singularity_image_path(obj, new_raw_native_path)
      if ischar(new_raw_native_path)
        new_path = apt.MetaPath(new_raw_native_path, 'native', 'universal') ;
        new_path = new_path.asWsl() ;
      elseif isa(new_raw_native_path, 'apt.Path')
        new_path = apt.MetaPath(new_raw_native_path, 'native', 'universal') ;
        new_path = new_path.asWsl() ;
      elseif isa(new_raw_native_path, 'apt.MetaPath')
        new_path = new_raw_native_path.asWsl() ;
      else
        error('APT:invalidValue', 'Invalid value for the Singularity image path');
      end
      obj.customSingularityImagePath_ = new_path ;
      obj.didOverrideDefaultSingularityImagePath_ = true ;
    end

    function result = get.singularity_image_path(obj)
      if obj.didOverrideDefaultSingularityImagePath_
        result = obj.customSingularityImagePath_ ;
      else
        result = DLBackEndClass.DEFAULT_SINGULARITY_IMAGE_PATH ;
      end
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

    % function set.condaEnv(obj, new_value)
    %   % Check for crazy values
    %   if ischar(new_value) && ~isempty(new_value) ,
    %     % all is well
    %   else
    %     error('APT:invalidValue', '"%s" is a not valid value for the conda environment', new_value);
    %   end        
    %   % Actually set the value
    %   obj.condaEnv = new_value ;
    % end    
  end  % methods block
 
  methods
    function [return_code, stdouterr] = runBatchCommandOutsideContainer_(obj, basecmd, varargin)
      % Run the basecmd using ShellCommand.run(), after wrapping suitably for the type of
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
      % command output. Note that any file paths in the bascmd must be *WSL* paths.
      % That's the main reason this method is private-by-convention.
      switch obj.type,
        case DLBackEnd.AWS
          % For AWS backend, use the AWSec2 method of the same name
          [return_code, stdouterr] = obj.awsec2.runBatchCommandOutsideContainer(basecmd, varargin{:}) ;
        case DLBackEnd.Bsub,
          % For now, we assume Matlab frontend is running on a JRC cluster node,
          % which means the filesystem is local.
          command = basecmd ;
          [return_code, stdouterr] = command.run('failbehavior', 'silent', 'verbose', false, varargin{:}) ;
        case DLBackEnd.Conda
          command = basecmd ;
          [return_code, stdouterr] = command.run('failbehavior', 'silent', 'verbose', false, varargin{:}) ;
        case DLBackEnd.Docker
          % Even if docker host is remote, we assume all files we need to access are on the
          % same path on the remote host.
          command = basecmd ;
          [return_code, stdouterr] = command.run('failbehavior', 'silent', 'verbose', false, varargin{:}) ;
        otherwise
          error('Not implemented: %s',obj.type);
      end
      % Things passed in with varargin should overide things we set here
    end  % function

    % function result = remoteMoviePathFromLocal(obj, localPath)
    %   % Convert a local movie path to the remote equivalent.
    %   % For non-AWS backends, this is the identity function.
    %   if isequal(obj.type, DLBackEnd.AWS) ,
    %     result = AWSec2.remoteMoviePathFromLocal(localPath) ;
    %   else
    %     result = localPath ;
    %   end
    % end

    % function result = remoteMoviePathsFromLocal(obj, localPathFromMovieIndex)
    %   % Convert a cell array of local movie paths to their remote equivalents.
    %   % For non-AWS backends, this is the identity function.
    %   if isequal(obj.type, DLBackEnd.AWS) ,
    %     result = AWSec2.remoteMoviePathsFromLocal(localPathFromMovieIndex) ;
    %   else
    %     result = localPathFromMovieIndex ;
    %   end
    % end

    function uploadMovies(obj, nativePathFromMovieIndex)
      % Upload movies to the backend, if necessary.
      
      % Validate input
      assert(iscell(nativePathFromMovieIndex), 'nativePathFromMovieIndex must be a cell array');
      
      % Local function to convert char to MetaPath
      function result = convertToMetaPath(path)
        if ischar(path)
          result = apt.MetaPath(path, apt.PathLocale.native, apt.FileRole.movie);
        else
          result = path;
        end
      end
      
      % Convert char elements to native MetaPaths
      nativePathFromMovieIndex = cellfun(@convertToMetaPath, nativePathFromMovieIndex, 'UniformOutput', false);
      
      % Validate that all elements are native movie MetaPaths
      cellfun(@(path) assert(isa(path, 'apt.MetaPath') && ...
        path.locale == apt.PathLocale.native && ...
        path.role == apt.FileRole.movie, ...
        'All elements must be native movie MetaPaths'), nativePathFromMovieIndex);
      
      if isequal(obj.type, DLBackEnd.AWS) ,
        % Convert to WSL MetaPaths
        wslPathFromMovieIndex = cellfun(@(path) path.asWsl(), nativePathFromMovieIndex, 'UniformOutput', false);
        obj.awsec2.uploadMovies(wslPathFromMovieIndex) ;
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
        % obj.stopEc2InstanceIfNeeded_() ;
          % We don't stop the ec2 instance anymore.  The aptAutoShutdown alarm will shut
          % it down after two hours of inactivity.  The instance takes roughly 15 min to
          % shutdown fully.  (The call to shut it down is async, and returns
          % immediately, but if you want to restart it later you have to wait for it to
          % fully shutdown first.)
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
      obj.jrcsimplebindpaths = true ;
      
      % In modern versions, we always have a .awsec2, whether we need it or not
      if isempty(obj.awsec2) ,
        obj.awsec2 = AWSec2() ;
      end
      
      % Modernize the AWSec2 object
      obj.awsec2.modernize();

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

      [dockerimgfull, minFreeMem, condaEnv, verbose] = ...
        myparse(varargin,...
                'dockerimg',obj.dockerimgfull,...
                'minfreemem',obj.minFreeMem,...
                'condaEnv',obj.condaEnv,...
                'verbose',0) ;  %#ok<PROPLC>
      
      gpuid = [];
      freemem = 0;
      gpuInfo = [];
      aptDeepnetPathNative = apt.MetaPath(APT.getpathdl(), apt.PathLocale.native, apt.FileRole.source) ;  % native path
      aptDeepnetPathWsl = aptDeepnetPathNative.asWsl() ;  % WSL path
      
      switch obj.type,
        case DLBackEnd.Docker
          baseCommand = apt.ShellCommand({'echo', 'START', ';', 'python', 'parse_nvidia_smi.py', ';', 'echo', 'END'}, ...
                                         apt.PathLocale.wsl, apt.Platform.posix) ;
          bindPath = {aptDeepnetPathWsl}; % don't use guarded
          command = wrapCommandDocker(baseCommand,...
                                      'dockerimg',dockerimgfull,...
                                      'containername','aptTestContainer',...
                                      'bindpath',bindPath,...
                                      'detach',false);
          [st,res] = command.run('verbose', verbose);
          if st ~= 0,
            warning('Error getting GPU info: %s\n%s',res,command);
            if ispc() && contains(res, 'WSL')
              fprintf('\nOn Windows, Docker Desktop has to be running for the Docker backend to work.\n\n')
            end
            return
          end
        case DLBackEnd.Conda
          scriptPath = aptDeepnetPathWsl.append('parse_nvidia_smi.py') ;
          baseCommand = apt.ShellCommand({'echo', 'START', '&&', 'python', scriptPath, '&&', 'echo', 'END'}, ...
                                         apt.PathLocale.wsl, ...
                                         apt.Platform.posix) ;
          command = wrapCommandConda(baseCommand, 'condaEnv', condaEnv) ;
          [st,res] = command.run() ;
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
    
    function r = aptSourceDirRootRemoteAsChar_(obj)
      % Returns the full path to the remote copy of the APT source, as a char array.
      switch obj.type
        case DLBackEnd.Bsub
          r = APT.Root;
        case DLBackEnd.AWS
          r = char(AWSec2.remoteAPTSourceRootDir) ;
        case DLBackEnd.Docker
          r = APT.Root;
        case DLBackEnd.Conda
          r = APT.Root;
      end
      assert(ischar(r)) ;
    end

    % function r = getAPTDeepnetRoot(obj)
    %   r = [obj.aptSourceDirRoot '/deepnet'];
    % end
        
    function tfSucc = writeCmdToFile(obj, syscmds, cmdfiles, jobdesc)  % const method
      % Write each syscmds{i} to each cmdfiles{i}, on the filesystem where the
      % commands will be executed. syscmds should be a scalar ShellCommand object
      % or a cell array of ShellCommand objects.  cmdfiles should be a MetaPath
      % object or a cell array of MetaPath objects, all with locale wsl.
      if nargin < 4,
        jobdesc = 'job';
      end
      if isa(syscmds, 'apt.ShellCommand')
        syscmds = {syscmds};
      end
      if isa(cmdfiles, 'apt.MetaPath') 
        cmdfiles = {cmdfiles};
      end
      tfSucc = false(1,numel(syscmds));
      assert(numel(cmdfiles) == numel(syscmds));
      for i = 1:numel(syscmds),
        syscmd = syscmds{i} ;
        cmdfile = cmdfiles{i} ;
        syscmdWithNewline = sprintf('%s\n', syscmd.char()) ;
        try 
          obj.writeStringToFile_(cmdfile, syscmdWithNewline) ;
          didSucceed = true ;
          errorMessage = '' ;
        catch me
          didSucceed = false ;
          errorMessage = me.message ;
        end
        tfSucc(i) = didSucceed ;
        if didSucceed ,
          fprintf('Wrote command for %s %d to cmdfile %s.\n',jobdesc,i,cmdfile);
        else
          warningNoTrace(errorMessage);
        end
      end  % for
    end  % function
  end  % methods block
  
  % methods % Bsub
  %   function aptroot = bsubSetRootUpdateRepo_(obj)
  %     aptroot = apt.bsubSetRootUpdateRepo(obj.deepnetrunlocal) ;
  %     obj.bsubaptroot = aptroot;
  %   end
  % end  % methods
  
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
      command0 = apt.ShellCommand({dockercmd, 'version', '--format', fmtspec}, apt.PathLocale.wsl, apt.Platform.posix);
      if ~isempty(obj.dockerremotehost),
        command = wrapCommandSSH(command0,'host',obj.dockerremotehost);
      else
        command = command0;
      end
      
      tfsucc = false;
      clientver = '';
      clientapiver = '';
        
      [st,res] = command.run();
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

    % function scpUploadOrVerify(obj, varargin)
    %   if isequal(obj.type, DLBackEnd.AWS) ,
    %     aws = obj.awsec2;
    %     aws.scpUploadOrVerify(varargin{:}) ;
    %   end      
    % end

    % function rsyncUpload(obj, src, dest)
    %   if isequal(obj.type, DLBackEnd.AWS) ,
    %     aws = obj.awsec2 ;
    %     aws.rsyncUpload(src, dest) ;
    %   end      
    % end
    
  end  % public methods block

  methods
    function [didsucceed, message] = mkdir_(obj, nativeFolderPath)
      % Create the named directory, either locally or remotely, depending on the
      % backend type.

      wslFolderPath = nativeFolderPath.asWsl() ;
      if obj.type == DLBackEnd.AWS ,
        [didsucceed, message] = obj.awsec2.mkdir(wslFolderPath) ;
      else
        baseCommand = apt.ShellCommand({'mkdir', '-p', wslFolderPath}, apt.PathLocale.wsl, apt.Platform.posix) ;
        [status, message] = obj.runBatchCommandOutsideContainer_(baseCommand) ;
        didsucceed = (status==0) ;
      end
    end  % function

    function [didsucceed, msg] = deleteFile(obj, nativeFilePath)
      % Delete the named file, either locally or remotely, depending on the
      % backend type.
      assert(isa(nativeFilePath, 'apt.MetaPath'), 'nativeFilePath must be an apt.MetaPath');
      assert(nativeFilePath.locale == apt.PathLocale.native, 'nativeFilePath must have native locale');
      wslFileMetaPath = nativeFilePath.asWsl() ;
      base_command = apt.ShellCommand({'rm', '-f', wslFileMetaPath}, apt.PathLocale.wsl, apt.Platform.posix) ;
      [status, msg] = obj.runBatchCommandOutsideContainer_(base_command) ;
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

    function writeStringToCacheFile(obj, nativeFilePathAsChar, str)
      % Write the given string to a file in the cache, overrwriting any previous contents.
      % nativeFilePathAsChar should be a native path, represented as a charray.
      % Throws if unable to write string to file.

      % Validate input
      assert(ischar(nativeFilePathAsChar) && (isempty(nativeFilePathAsChar) || isrow(nativeFilePathAsChar))) ;
      assert(ischar(str) && (isempty(str) || isrow(str))) ;

      nativeFilePath = apt.MetaPath(nativeFilePathAsChar, apt.PathLocale.native, apt.FileRole.cache) ;
      obj.writeStringToFile_(nativeFilePath, str) ;  % throws if unable to write file
    end  % function
    
    function writeStringToFile_(obj, filePath, str)
      % Write the given string to a file, overrwriting any previous contents.
      % filePath should be a native or wsl MetaPath.
      % Throws if unable to write string to file.

      % Validate input
      assert(isa(filePath, 'apt.MetaPath'), 'filePath must be an apt.MetaPath');
      assert(filePath.locale == apt.PathLocale.native || filePath.locale == apt.PathLocale.wsl, ...
             'filePath must have native or WSL locale');

      if isequal(obj.type, DLBackEnd.AWS)
        % AWS backend expects WSL MetaPath
        wslMetaPath = filePath.asWsl();
        obj.awsec2.writeStringToFile(wslMetaPath, str);
      else
        % Local filesystem - convert to native string
        nativeMetaPath = filePath.asNative();
        nativeMetaPathAsChar = nativeMetaPath.charUnescaped() ;
        fo = file_object(nativeMetaPathAsChar, 'w');
        fo.fprintf('%s', str);
      end
    end  % function

    function updateRepo(obj)
      % Update the APT repo on the backend.  While we're at it, make sure the
      % pretrained weights are downloaded.  The method formerly known as
      % setupForTrainingOrTracking().
      % localCacheDir should be e.g. /home/joeuser/.apt/tp662830c8_246a_49c6_816c_470db4ecd950
      % localCacheDir is not currently used, but will be needed to get the JRC
      % backend working properly for AD-linked Linux workstations.
      switch obj.type
        % case DLBackEnd.Bsub ,
        %   obj.bsubSetRootUpdateRepo_();
        case {DLBackEnd.Conda, DLBackEnd.Docker, DLBackEnd.Bsub} ,
          aptroot = APT.Root;
          apt.downloadPretrainedWeights('aptroot', aptroot) ;
        case DLBackEnd.AWS ,
          obj.awsec2.updateRepo() ;
        otherwise
          error('Unknown backend type') ;
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

    command = trainCodeGenBase(dmc,varargin)  % defined in own file
    command = trackCodeGenBase(totrackinfo, varargin)  % defined in own file
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
        % obj.training_logcmds_ = cell(0,1) ;
        obj.training_jobids_ = cell(0,1) ;
      else
        obj.tracking_syscmds_ = cell(0,1) ;
        obj.tracking_cmdfiles_ = cell(0,1) ;
        % obj.tracking_logcmds_ = cell(0,1) ;
        obj.tracking_jobids_ = cell(0,1) ;
      end
    end  % function

    function registerTrainingJob(obj, dmcjob, tracker, gpuids, do_just_generate_db)
      % Register a single training job with the backend, for later spawning via
      % spawnRegisteredJobs().

      % Get the root of the remote source tree
      remoteAptRootAsChar = obj.aptSourceDirRootRemoteAsChar_() ;
      
      ignore_local = (obj.type == DLBackEnd.Bsub) ;  % whether to pass the --ignore_local options to APTInterface.py
      basecmd = DLBackEndClass.trainCodeGenBase(dmcjob,...
                                                'ignore_local',ignore_local,...
                                                'nativeaptroot',APT.Root,...
                                                'do_just_generate_db',do_just_generate_db);
      args = obj.determineArgumentsForSpawningJob_(tracker,gpuids,dmcjob,remoteAptRootAsChar,'train');
      syscmd = obj.wrapCommandToBeSpawnedForBackend_(basecmd,args{:});
      commandFileNativePathAsChar = DeepModelChainOnDisk.getCheckSingle(dmcjob.trainCmdfileLnx());
      commandFileNativePath = apt.MetaPath(commandFileNativePathAsChar, apt.PathLocale.native, apt.FileRole.cache) ;
      commandFileWslPath = commandFileNativePath.asWsl() ;
      % logcmd = obj.generateLogCommand_('train', dmcjob) ;

      % Add all the commands to the registry
      obj.training_syscmds_{end+1,1} = syscmd ;
      % obj.training_logcmds_{end+1,1} = logcmd ;
      obj.training_cmdfiles_{end+1,1} = commandFileWslPath ;
      obj.training_jobids_{end+1,1} = [] ;  % indicates not-yet-spawned job
    end

    function registerTrackingJob(obj, totrackinfo, deeptracker, gpuids, track_type)
      % Register a single tracking job with the backend, for later spawning via
      % spawnRegisteredJobs().
      % track_type should be one of {'track', 'link', 'detect'}

      % Get the root of the remote source tree
      remoteAptRootAsChar = obj.aptSourceDirRootRemoteAsChar_() ;

      % totrackinfo has local paths, need to remotify them
      % remotetotrackinfo = totrackinfo.copy() ;
      % remotetotrackinfo.changePathsToRemoteFromWsl(obj.wslProjectCachePath, obj) ;
      remotetotrackinfo = obj.changeToTrackInfoPathsToRemoteFromWsl_(totrackinfo) ;

      ignore_local = (obj.type == DLBackEnd.Bsub) ;  % whether to pass the --ignore_local options to APTInterface.py
      basecmd = DLBackEndClass.trackCodeGenBase(totrackinfo,...
                                                'ignore_local',ignore_local,...
                                                'nativeaptroot',APT.Root,...
                                                'track_type',track_type);
      args = obj.determineArgumentsForSpawningJob_(deeptracker, gpuids, remotetotrackinfo, remoteAptRootAsChar, 'track') ;
      syscmd = obj.wrapCommandToBeSpawnedForBackend_(basecmd, args{:}) ;
      commandFilePathAsChar = DeepModelChainOnDisk.getCheckSingle(remotetotrackinfo.cmdfile) ;
      commandFileNativePath = apt.MetaPath(commandFilePathAsChar, apt.PathLocale.native, apt.FileRole.cache) ;
      commandFileWslPath = commandFileNativePath.asWsl() ;
      
      % logcmd = obj.generateLogCommand_('track', remotetotrackinfo) ;
    
      % Add all the commands to the registry
      obj.tracking_syscmds_{end+1,1} = syscmd ;
      % obj.tracking_logcmds_{end+1,1} = logcmd ;
      obj.tracking_cmdfiles_{end+1,1} = commandFileWslPath ;
      obj.tracking_jobids_{end+1,1} = [] ;  % indicates not-yet-spawned job
    end

    function spawnRegisteredJobs(obj, train_or_track, varargin)
      % Spawn all the training/tracking jobs that have been previously registered.
      % On entry, all jobs of the given type should be 
      [jobdesc, do_call_apt_interface_dot_py] = myparse( ...
        varargin, ...
        'jobdesc', 'job', ...
        'do_call_apt_interface_dot_py', true) ;

      % Sort out which registered jobs will be spawned.
      if strcmp(train_or_track, 'train') ,
        syscmds = obj.training_syscmds_ ;
        % logcmds = obj.training_logcmds_ ;
        cmdfiles = obj.training_cmdfiles_ ;
      elseif strcmp(train_or_track, 'track') ,
        syscmds = obj.tracking_syscmds_ ;
        % logcmds = obj.tracking_logcmds_ ;
        cmdfiles = obj.tracking_cmdfiles_ ;
      else
        error('DLBackEndClass:unknownJobType', 'The job type ''%s'' is not valid', train_or_track) ;
      end

      % Write the commands to files
      if ~isempty(cmdfiles),
        obj.writeCmdToFile(syscmds,cmdfiles,jobdesc);
      end

      % Actually spawn the jobs
      [didSpawnAllJobs, reason, spawned_jobids] = DLBackEndClass.spawnJobs(syscmds, obj.type, jobdesc, do_call_apt_interface_dot_py) ;
      % obj.ensureJobIsNotAlive(spawned_jobids{1}) ;  
      %   % USED FOR DEBUGGING, to simulate a failure of a spawned job without
      %   % production of an error file.

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
        % Now that stray jobs were killed, throw error
        error(reason) ;
      end  % if        
    end  % function
  end  % methods

  methods (Static)
    function [didSpawnAllJobs, reason, jobidFromJobIndex] = spawnJobs(syscmds, backend_type, jobdesc, do_call_apt_interface_dot_py)
      % Spawn the jobs specified in syscmds.  On return, didSpawnAllJobs indicates
      % whether all went well.  *Regardless of the value of didSpawnAllJobs,*
      % jobidFromJobIndex contains the jobids of all the jobs that were successfully
      % spawned.  
      jobCount = numel(syscmds) ;
      jobidFromJobIndex = cell(0,1) ;  % We only add jobids to this once they have successfully been spawned.
      for jobIndex = 1:jobCount ,
        syscmd = syscmds{jobIndex} ;
        fprintf('%s\n',syscmd.char());
        if do_call_apt_interface_dot_py ,
          [rc,stdouterr] = syscmd.run('failbehavior', 'silent');
          didSpawn = (rc == 0) ;          
          if didSpawn ,
            jobid = DLBackEndClass.parseJobID(backend_type, stdouterr);
          else
            didSpawnAllJobs = false ;
            reason = stdouterr ;
            return
          end
        else
          % Pretend it's a failure, for expediency.
          didSpawnAllJobs = false ;
          reason = 'It was requested that APT_interface.py not be called.' ;
          return
        end

        % If get here, this job spawn succeeded
        jobidFromJobIndex{jobIndex,1} = jobid ;  % Add to end, keeping it a col vector
        assert(ischar(jobid)) ;
        fprintf('%s %d spawned, ID = %s\n\n',jobdesc,jobIndex,jobid);
      end  % for      
      didSpawnAllJobs = true ;  % if get here, all is well
      reason = '' ;
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
      result = logical(cellfun(@(jobid)(obj.isJobAlive(jobid)), jobids)) ;  
        % boolean array
        % Need the call to logical() in case jobids is empty, because cellfun()
        % defaults to double output in this case.
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
      cmd = apt.ShellCommand({apt.dockercmd(), 'ps', '-q', '-f', sprintf('id=%s', jobidshort)}, apt.PathLocale.wsl, apt.Platform.posix);      
      [st,res] = obj.runBatchCommandOutsideContainer_(cmd);
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
      bjobsCommand = apt.ShellCommand({'bjobs', '-o', 'stat', '-noheader', jobid}, apt.PathLocale.remote, apt.Platform.posix);
      sshCommand = wrapCommandSSH(bjobsCommand,'host',DLBackEndClass.jrchost);
      [st,res] = sshCommand.run('failbehavior', 'silent', 'verbose', false) ;
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
      command = apt.ShellCommand({'/usr/bin/pgrep', '--pgroup', jobid}, apt.PathLocale.wsl, apt.Platform.posix) ;  % For conda backend, the jobid is a PGID
      [return_code, stdouterr] = command.run('failbehavior', 'silent') ;  %#ok<ASGLU>
      % pgrep exits with return_code == 1 if there is no such PGID.  Not great for
      % detecting when something *else* has gone wrong, but whaddayagonnado?
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
      bkillcmd0 = apt.ShellCommand({'bkill', jobid}, apt.PathLocale.remote, apt.Platform.posix);
      bkillcmd = wrapCommandSSH(bkillcmd0,'host',DLBackEndClass.jrchost);
      [st,res] = bkillcmd.run('failbehavior', 'silent', 'verbose', false) ;
      if st~=0 ,
        error('Error occurred when trying to kill bsub job %s: %s', jobid, res) ;
      end
    end  % function

    function killJobConda_(obj, jobid)  %#ok<INUSD> 
      pgid = jobid ;  % conda backend uses PGID as the job id
      command = apt.ShellCommand({'kill', '--', ['-' pgid]}, apt.PathLocale.wsl, apt.Platform.posix) ;  % kill all processes in the process group
      [st, res] = command.run() ;
      if st ~= 0
        error('Failed to kill process group %s: %s', pgid, res);
      end
    end  % function

    function killJobDockerOrAWS_(obj, jobid)
      % Kill the docker job with job id jobid.
      % jobid is assumed to be a single job id, represented as an old-style string.      
      % Errors if no such job exists.
      cmd = apt.ShellCommand({apt.dockercmd(), 'kill', jobid}, apt.PathLocale.wsl, apt.Platform.posix);        
      [st,res] = obj.runBatchCommandOutsideContainer_(cmd) ;
        % It uses the docker executable, but it still runs outside the docker
        % container.
      if st~=0 ,
        error('Error occurred when trying to kill %s job %s: %s', char(obj.type), jobid, res) ;
      end
    end  % function

    function downloadTrackingFilesIfNecessary(obj, res, movfiles)
      % Errors if something goes wrong.
      if obj.type == DLBackEnd.AWS ,
        obj.awsec2.downloadTrackingFilesIfNecessary(res, movfiles) ;
      elseif obj.type == DLBackEnd.Bsub ,
        % Hack: For now, just wait a bit, to let (hopefully) NFS sync up
        pause(10) ;
      elseif obj.type == DLBackEnd.Conda ,
        % do nothing
      elseif obj.type == DLBackEnd.Docker ,
        if ~isempty(obj.dockerremotehost) ,
          % This path is for when the docker backend is running on a remote host.
          % Hack: For now, just wait a bit, to let (hopefully) NFS sync up.
          pause(10) ;
        end          
      else
        error('Internal error: Unknown DLBackEndClass type') ;
      end
    end  % function    

    function [tfsucc,res] = batchPoll(obj, nativeFsPollArgs)
      % fspollargs: [n] cellstr eg {'exists' '/my/file' 'existsNE' '/my/file2'}
      %
      % res: [n] cellstr of fspoll responses

      if obj.type == DLBackEnd.AWS ,
        wsl_fspollargs = wsl_fspollargs_from_native(nativeFsPollArgs) ;
        [tfsucc,res] = obj.awsec2.batchPoll(wsl_fspollargs) ;
      else
        error('Not implemented') ;        
        %fspoll_script_path = linux_fullfile(APT.Root, 'matlab/misc/fspoll.py') ;
      end
    end  % function

    function result = tfDoesCacheFileExist(obj, nativeFilePathAsChar)
      % Checks if the named cache file exists.
      assert(ischar(nativeFilePathAsChar)) ;
      nativeFilePath = apt.MetaPath(nativeFilePathAsChar, apt.PathLocale.native, apt.FileRole.cache) ;
      result = obj.fileExists_(nativeFilePath) ;
    end
    
    function result = fileExists_(obj, nativeFilePath)
      % Returns true iff the named file exists.
      % Should be consolidated with exist(), probably.  Note, though, that probably
      % need to be careful about checking for the file inside/outside the container.
      assert(isa(nativeFilePath, 'apt.MetaPath')) ;
      if obj.type == DLBackEnd.AWS ,
        wslFilePath = nativeFilePath.asWsl() ;
        result = obj.awsec2.fileExists(wslFilePath) ;
      else
        nativeFilePathAsChar = nativeFilePath.charUnescaped() ;
        result = logical(exist(nativeFilePathAsChar, 'file')) ;
      end
    end  % function

    function result = tfCacheFileExistsAndIsNonempty(obj, nativeFilePathAsChar)
      % Returns true iff the named file exists and is not zero-length.
      assert(ischar(nativeFilePathAsChar)) ;
      if obj.type == DLBackEnd.AWS ,
        nativeFilePath = apt.MetaPath(nativeFilePathAsChar, apt.PathLocale.native, apt.FileRole.cache) ;
        wslFilePath = nativeFilePath.asWsl() ;
        result = obj.awsec2.fileExistsAndIsNonempty(wslFilePath) ;
      else
        result = localFileExistsAndIsNonempty(nativeFilePathAsChar) ;
      end
    end  % function

    % function result = fileExistsAndIsGivenSize(obj, nativeFilePath, sz)
    %   % Returns true iff the named file exists and is the given size (in bytes).
    %   assert(isa(nativeFilePath, 'apt.MetaPath')) ;
    %   if obj.type == DLBackEnd.AWS ,
    %     wslFilePath = nativeFilePath.asWsl() ;
    %     result = obj.awsec2.fileExistsAndIsGivenSize(wslFilePath, sz) ;
    %   else
    %     result = localFileExistsAndIsGivenSize(nativeFilePath.char(), sz) ;
    %   end
    % end  % function

    function result = cacheFileContents(obj, nativeFilePathAsChar)
      % Return the contents of the named file, as an old-style string.
      % The behavior of this function when the file does not exist is kinda weird.
      % It is the way it is b/c it's designed for giving something helpful to
      % display in the monitor window.      
      assert(isa(nativeFilePathAsChar, 'char')) ;
      if obj.type == DLBackEnd.AWS ,
        nativeFilePath = apt.MetaPath(nativeFilePathAsChar, apt.PathLocale.native, apt.FileRole.cache) ;
        wslFilePath = nativeFilePath.asWsl() ;
        result = obj.awsec2.fileContents(wslFilePath) ;
      else
        if exist(nativeFilePathAsChar,'file') ,
          lines = readtxtfile(nativeFilePathAsChar);
          result = sprintf('%s\n',lines{:});
        else
          result = '<file does not exist>';
        end
      end
    end  % function
    
    % function lsdir(obj, native_dir_path)
    %   % List the contents of directory dir.  Contents just go to stdout, nothing is
    %   % returned.
    %   if obj.type == DLBackEnd.AWS ,
    %     wsl_dir_path = wsl_path_from_native(native_dir_path) ;
    %     obj.awsec2.lsdir(wsl_dir_path) ;
    %   else
    %     if ispc()
    %       lscmd = 'dir';
    %     else
    %       lscmd = 'ls -al';
    %     end
    %     cmd = sprintf('%s "%s"',lscmd,native_dir_path);
    %     system(cmd);
    %   end
    % end  % function

    function result = cacheFileModTime(obj, nativeFilePathAsChar)
      % Return the file-modification time (mtime) of the given file.  For an AWS
      % backend, this is the file modification time in seconds since epoch.  For
      % other backends, it's a Matlab datenum of the mtime.  So these should not be
      % compared across backend types.
      assert(isa(nativeFilePathAsChar, 'char')) ;
      if obj.type == DLBackEnd.AWS ,
        nativeFilePath = apt.MetaPath(nativeFilePathAsChar, apt.PathLocale.native, apt.FileRole.cache) ;        
        wslFilePath = nativeFilePath.asWsl() ;
        result = obj.awsec2.remoteFileModTime(wslFilePath) ;
      else
        dir_struct = dir(nativeFilePathAsChar) ;
        result = dir_struct.datenum ;
      end
    end  % function

    function suitcase = packParfevalSuitcase(obj)
      % Use before calling parfeval, to restore Transient properties that we want to
      % survive the parfeval boundary.
      suitcase = struct() ;
      suitcase.training_jobids_ = obj.training_jobids_ ;
      suitcase.tracking_jobids_ = obj.tracking_jobids_ ;
      if obj.type == DLBackEnd.AWS ,
        suitcase.awsec2 = obj.awsec2.packParfevalSuitcase() ;
      else
        suitcase.awsec2 = [] ;
      end
    end  % function
    
    function restoreAfterParfeval(obj, suitcase)
      % Should be called in background tasks run via parfeval, to restore fields that
      % should not be restored from persistence, but we want to survive the parfeval
      % boundary.
      obj.doesOwnResources_ = false ;  % don't want to free any resources on delete()
      obj.training_jobids_ = suitcase.training_jobids_ ;
      obj.tracking_jobids_ = suitcase.tracking_jobids_ ;
      if obj.type == DLBackEnd.AWS ,
        obj.awsec2.restoreAfterParfeval(suitcase.awsec2) ;
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

    function nframes = readTrkFileStatus(obj, nativeFilePathAsChar, isTextFile, logger)
      % Read the number of frames remaining according to the remote file
      % corresponding to absolute local file path
      % localFilepath.  If partFileIsTextStatus is true, this file is assumed to be a
      % text file.  Otherwise, it is assumed to be a .mat file.  If the file does
      % not exist or there's some problem reading the file, returns nan.
      if ~exist('isTextFile', 'var') || isempty(isTextFile) ,
        isTextFile = false ;
      end
      if ~exist('logger', 'var') || isempty(logger) ,
        logger = FileLogger(1, 'DLBackEndClass::readTrkFileStatus()') ;
      end

      if obj.type == DLBackEnd.AWS ,
        % AWS backend
        nativeFileMetaPath = apt.MetaPath(nativeFilePathAsChar, apt.PathLocale.native, apt.FileRole.cache);
        wslFileMetaPath = nativeFileMetaPath.asWsl();
        nframes = obj.awsec2.readTrkFileStatus(wslFileMetaPath, isTextFile, logger) ;
      else
        % If non-AWS backend
        if ~exist(nativeFilePathAsChar,'file'),
          nframes = nan ;
          return
        end
        if isTextFile ,
          s = obj.cacheFileContents(nativeFilePathAsChar) ;
          nframes = TrkFile.getNFramesTrackedTextFile(s) ;
        else
          try
            nframes = TrkFile.getNFramesTrackedMatFile(nativeFilePathAsChar) ;
          catch
            fprintf('Could not read tracking progress from %s\n',nativeFilePathAsChar);
            nframes = nan ;
          end
        end        
      end
      % if isnan(nframes) ,
      %   nop() ;
      % end
    end  % function
    
    % function cmdfull = wrapCommandSSHAWS(obj, cmdremote, varargin)
    %   cmdfull = obj.awsec2.wrapCommandSSH(cmdremote, varargin{:}) ;
    % end

    function maxiter = getMostRecentModel(obj, dmc)  % constant method
      % Get the number of iterations completed for the model indicated by dmc.
      % Note that dmc will have native paths in it.
      % Also note that maxiter is in general a row vector.      
      if obj.type == DLBackEnd.AWS ,
        maxiter = obj.awsec2.getMostRecentModel(dmc) ;
      else
        maxiter = DLBackEndClass.getMostRecentModelLocal_(dmc) ;
      end
    end  % function
  end  % methods

  methods (Static)
    function maxiter = getMostRecentModelLocal_(dmc)
      % Get the number of iterations completed for the model indicated by dmc.
      % Note that dmc will have native paths in it.
      % Also note that maxiter is in general a row vector.
      % This method should only be called when you know the *local* dmc is the
      % up-to-date one.  Hence the underscore.
      [modelglob,idx] = dmc.trainModelGlob();
      [dirModelChainLnx] = dmc.dirModelChainLnx(idx);

      maxiter = nan(1,numel(idx));
      for i = 1:numel(idx),
        modelfiles= mydir(fullfile(dirModelChainLnx{i},modelglob{i}));
        if isempty(modelfiles),
          continue;
        end
        for j = 1:numel(modelfiles),
          iter = DeepModelChainOnDisk.getModelFileIter(modelfiles{j});
          if ~isempty(iter),
            maxiter(i) = max(maxiter(i),iter);
          end
        end
      end
    end  % function
  end  % methods (Static)

  methods
    function uploadProjectCacheIfNeeded(obj, nativeProjectCachePath)
      if obj.type == DLBackEnd.AWS ,
         nativeProjectCacheMetaPath = apt.MetaPath(nativeProjectCachePath, apt.PathLocale.native, apt.FileRole.cache);
         wslProjectCacheMetaPath = nativeProjectCacheMetaPath.asWsl();
         obj.awsec2.uploadProjectCacheIfNeeded(wslProjectCacheMetaPath) ;
      end
    end

    function downloadProjectCacheIfNeeded(obj, nativeCacheDirPath)
      % If the model chain is remote, download it
      if obj.type == DLBackEnd.AWS ,
         nativeCacheDirMetaPath = apt.MetaPath(nativeCacheDirPath, apt.PathLocale.native, apt.FileRole.cache);
         wslCacheDirMetaPath = nativeCacheDirMetaPath.asWsl();
         obj.awsec2.downloadProjectCacheIfNeeded(wslCacheDirMetaPath) ;
      end
    end  % function

    function result = get.isProjectCacheRemote(obj)
      result = (obj.type == DLBackEnd.AWS) && obj.awsec2.isProjectCacheRemote ;
    end  % function

    function result = get.isProjectCacheLocal(obj)
      result = ~obj.isProjectCacheRemote ;
    end  % function

    function prepareFilesForTracking(backend, toTrackInfo)
      backend.ensureFoldersNeededForTrackingExist_(toTrackInfo) ;
      backend.ensureCacheFilesDoNotExist_({toTrackInfo.getErrfile()}, 'error file') ;
      backend.ensureCacheFilesDoNotExist_(toTrackInfo.getPartTrkFiles(), 'partial tracking result') ;
      %backend.ensureFilesDoNotExist_({toTrackInfo.getKillfile()}, 'kill files') ;
    end  % function

    function ensureFoldersNeededForTrackingExist_(obj, toTrackInfo)
      % Paths in toTrackInfo are native paths
      nativeDirPathAsCharFromIndex = toTrackInfo.trkoutdir ;
      desc = 'trk cache dir' ;
      for i = 1:numel(nativeDirPathAsCharFromIndex) ,
        nativeDirPathAsChar = nativeDirPathAsCharFromIndex{i} ;
        nativeDirPath = apt.MetaPath(nativeDirPathAsChar, apt.PathLocale.native, apt.FileRole.cache) ;
        if ~obj.fileExists_(nativeDirPath) ,
          [succ, msg] = obj.mkdir_(nativeDirPath) ;
          if succ
            fprintf('Created %s: %s\n', desc, nativeDirPathAsChar) ;
          else
            error('Failed to create %s %s: %s', desc, nativeDirPathAsChar, msg) ;
          end
        end
      end
    end  % function

    function ensureCacheFilesDoNotExist_(obj, nativeFilePathAsCharFromIndex, desc)
      % native_file_paths are, umm, native paths
      for i = 1:numel(nativeFilePathAsCharFromIndex) ,
        nativeFilePathAsChar = nativeFilePathAsCharFromIndex{i} ;
        nativeFilePath = apt.MetaPath(nativeFilePathAsChar, apt.PathLocale.native, apt.FileRole.cache) ;
        if obj.fileExists_(nativeFilePath) ,
          fprintf('Deleting %s %s', desc, nativeFilePathAsChar) ;
          obj.deleteFile(nativeFilePath);
        end
        if obj.fileExists_(nativeFilePath),
          error('Failed to delete %s: file still exists',nativeFilePathAsChar);
        end
      end
    end  % function

    function result = get.wslProjectCachePath(obj)
      % The local DMC root dir, as a WSL path.
      result = obj.awsec2.wslProjectCachePath ;
    end  % function

    function result = get.nativeProjectCachePath(obj)
      % The local DMC root dir, as a WSL path.
      wslPath = obj.awsec2.wslProjectCachePath ;
      result = wslPath.asNative() ;
    end  % function

    function set.wslProjectCachePath(obj, value) 
      % Set the WSL project cache path. Accepts either a char or a WSL MetaPath.
      % Converts to WSL MetaPath before passing to obj.awsec2.
      if ischar(value) || isstring(value)
        wslMetaPath = apt.MetaPath(char(value), apt.PathLocale.wsl, apt.FileRole.cache);
      elseif isa(value, 'apt.MetaPath')
        assert(value.locale == apt.PathLocale.wsl, 'MetaPath must have WSL locale');
        wslMetaPath = value;
      else
        error('wslProjectCachePath must be a char, string, or WSL apt.MetaPath');
      end
      
      obj.awsec2.wslProjectCachePath = wslMetaPath;
    end  % function

    function set.nativeProjectCachePath(obj, value) 
      % Set the local DMC root dir. Accepts either a char or a native MetaPath.
      % Converts to WSL MetaPath before passing to obj.awsec2.
      if ischar(value) || isstring(value)
        nativeMetaPath = apt.MetaPath(char(value), apt.PathLocale.native, apt.FileRole.cache);
      elseif isa(value, 'apt.MetaPath')
        assert(value.locale == apt.PathLocale.native, 'MetaPath must have native locale');
        nativeMetaPath = value;
      else
        error('nativeProjectCachePath must be a char, string, or native apt.MetaPath');
      end
      
      wslMetaPath = nativeMetaPath.asWsl();
      obj.awsec2.wslProjectCachePath = wslMetaPath;
    end  % function

    function result = get.remoteDMCRootDir(obj)  %#ok<MANU>
      % Get the remote DMC root dir.  Returned as a remote path.
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

    % function set.awsInstanceType(obj, value)
    %   obj.awsec2.instanceType = value ;
    % end  % function
    
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
    
    % function logcmd = generateLogCommand_(obj, train_or_track, dmcjob_or_totrackinfojob)  % constant method
    %   if strcmp(train_or_track, 'train') ,
    %     dmcjob = dmcjob_or_totrackinfojob ;
    %     if obj.type == DLBackEnd.Docker ,
    %       containerName = DeepModelChainOnDisk.getCheckSingle(dmcjob.trainContainerName) ;
    %       native_log_file_path = DeepModelChainOnDisk.getCheckSingle(dmcjob.trainLogLnx) ;
    %       logcmd = obj.generateLogCommandForDockerBackend_(containerName, native_log_file_path) ;
    %     else
    %       logcmd = '' ;
    %     end
    %   elseif strcmp(train_or_track, 'track') ,
    %     totrackinfojob = dmcjob_or_totrackinfojob ;
    %     if obj.type == DLBackEnd.Docker ,
    %       containerName = totrackinfojob.containerName ;
    %       native_log_file_path = totrackinfojob.logfile ;
    %       logcmd = obj.generateLogCommandForDockerBackend_(containerName, native_log_file_path) ;
    %     else
    %       logcmd = '' ;
    %     end
    %   else
    %     error('train_or_track had illegal value ''%s''', train_or_track) ;
    %   end
    % end  % function

    % function cmd = generateLogCommandForDockerBackend_(backend, containerName, native_log_file_path)  % constant method
    %   assert(backend.type == DLBackEnd.Docker);
    %   dockercmd = apt.dockercmd();
    %   wsl_log_file_path = wsl_path_from_native(native_log_file_path) ;
    %   cmd = ...
    %     sprintf('%s logs -f %s &> %s', ... 
    %             dockercmd, ...
    %             containerName, ...
    %             escape_string_for_bash(wsl_log_file_path)) ;
    %   is_docker_remote = ~isempty(backend.dockerremotehost) ;
    %   if is_docker_remote
    %     cmd = wrapCommandSSH(cmd,'host',backend.dockerremotehost);
    %   end
    %   cmd = sprintf('%s &', cmd);
    % end  % function

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
      cmd = apt.ShellCommand({apt.dockercmd(), 'ps', '--filter', sprintf('id=%s', jobidshort)}, apt.PathLocale.wsl, apt.Platform.posix) ;      
      [rc, stdouterr] = obj.runBatchCommandOutsideContainer_(cmd) ;
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
      bjobsCommand = apt.ShellCommand({'bjobs', jobid}, apt.PathLocale.remote, apt.Platform.posix) ;
      sshCommand = wrapCommandSSH(bjobsCommand, 'host', DLBackEndClass.jrchost) ;
        % For the bsub backend, obj.runBatchCommandOutsideContainer() still runs
        % things locally, since that's what you want for e.g. commands that check on
        % file status.
      [rc, stdouterr] = obj.runBatchCommandOutsideContainer_(sshCommand) ;
      if rc==0 ,
        result = stdouterr ;
      else
        result = sprintf('Error occurred when checking status of bsub job %s: %s', jobid, stdouterr) ;
      end
    end  % function

    function result = detailedStatusStringConda_(obj, jobid)  %#ok<INUSD>
      % Returns true if there is a running conda job with ID jobid.
      % jobid is assumed to be a single job id, represented as an old-style string.      
      command = apt.ShellCommand({'/usr/bin/pgrep', '--pgroup', jobid}, apt.PathLocale.wsl, apt.Platform.posix) ;  % For conda backend, the jobid is a PGID
      [return_code, stdouterr] = command.run('failbehavior', 'silent') ;
      % pgrep exits with return_code == 1 if there is no such PGID.  Not great for
      % detecting when something *else* has gone wrong, but whaddayagonnado?
      % We capture stdouterr to prevent it getting spit out to the Matlab console.
      % We use a variable name instead of ~ in case we need to debug in here at some
      % point.
      if return_code==0 ,
        result = stdouterr ;
      else
        result = sprintf('Error occurred when checking status of conda job %s: %s', jobid, stdouterr) ;
      end
    end  % function


    
    
  end  % methods

  methods
    function cmd = wrapCommandToBeSpawnedForBackend_(obj, basecmd, varargin)  % const method
      switch obj.type,
        case DLBackEnd.AWS
          cmd = wrapCommandToBeSpawnedForAWSBackend_(obj, basecmd, varargin{:});
        case DLBackEnd.Bsub,
          cmd = wrapCommandToBeSpawnedForBsubBackend_(obj, basecmd, varargin{:});
        case DLBackEnd.Conda
          cmd = wrapCommandToBeSpawnedForCondaBackend_(obj, basecmd, varargin{:});
        case DLBackEnd.Docker
          cmd = wrapCommandToBeSpawnedForDockerBackend_(obj, basecmd, varargin{:});
        otherwise
          error('Not implemented: %s',obj.type);
      end
    end
    
    function result = wrapCommandToBeSpawnedForAWSBackend_(obj, basecmd, varargin)  % const method
      % Wrap for docker, returns Linux/WSL-style command string
      
      % Parse arguments
      [dockerargs, sshargs] = ...
        myparse_nocheck(varargin, ...
                        'dockerargs',{}, ...
                        'sshargs',{}) ;
    
      % Wrap for docker
      dockerimg = 'bransonlabapt/apt_docker:apt_20230427_tf211_pytorch113_ampere' ;
      homePathWsl = apt.MetaPath('/home', 'wsl', 'slashhome') ;
      bindpath = {homePathWsl} ;
      codestr = ...
        wrapCommandDocker(basecmd, ...
                          'dockerimg',dockerimg, ...
                          'bindpath',bindpath, ...
                          dockerargs{:}) ;
    
      % Wrap for ssh'ing into a remote docker host, if needed
      %result = obj.wrapCommandSSHAWS(codestr, sshargs{:}) ;
      result = obj.awsec2.wrapCommandSSH(codestr, sshargs{:}) ;
    end
    
    function cmd = wrapCommandToBeSpawnedForBsubBackend_(obj, baseCommand, varargin)  % const method
      assert(isa(baseCommand, 'apt.ShellCommand'), 'baseCommand must be an apt.ShellCommand object');
      assert(baseCommand.tfDoesMatchLocale(apt.PathLocale.wsl), 'baseCommand must have wsl locale');

      % Parse optional args
      [singargs,bsubargs,sshargs] = myparse(varargin,'singargs',{},'bsubargs',{},'sshargs',{});

      % Wrap, wrap, wrap
      apptainerCommand = wrapCommandSing(baseCommand,singargs{:});
      bsubWslCommand = wrapCommandBsub(apptainerCommand,bsubargs{:});
    
      % already on cluster?
      tfOnCluster = ~isempty(getenv('LSB_DJOB_NUMPROC'));
      if tfOnCluster,
        % The Matlab environment vars cause problems with e.g. PyTorch
        cmd = prependStuffToClearMatlabEnvironment(bsubWslCommand) ;
      else
        % Doing ssh does not pass Matlab envars, so they don't cause problems in this case.  
        bsubRemoteCommand = obj.convertWslShellCommandToRemote(bsubWslCommand) ;
        cmd = wrapCommandSSH(bsubRemoteCommand,'host',DLBackEndClass.jrchost,sshargs{:});
      end
    end  % function
    
    function result = wrapCommandToBeSpawnedForCondaBackend_(obj, baseCommand, varargin)  % const method
      % Take a base command and run it in a conda env
      condafiedCommand = wrapCommandConda(baseCommand, 'condaEnv', obj.condaEnv, 'logfile', '/dev/null', 'gpuid', obj.gpuids(1), varargin{:}) ;
      nilCommand = apt.ShellCommand({}, condafiedCommand.locale, condafiedCommand.platform) ;
      result = nilCommand.cat('{', condafiedCommand, '&', '}', '&&', 'echo', '$!') ;  % echo $! to get the PID
    end
    
    function codestr = wrapCommandToBeSpawnedForDockerBackend_(obj, basecmd, varargin)  % const method
      % Take a base command and run it in a docker img
    
      % Determine the fallback gpuid, keeping in mind that backend.gpuids may be
      % empty.
      if isempty(obj.gpuids) ,
        fallback_gpuid = 1 ;
      else
        fallback_gpuid = obj.gpuids(1) ;
      end    
    
      % Call main function, returns Linux/WSL-style command string
      codestr = ...
        wrapCommandDocker(basecmd, ...
                          'dockerimg',obj.dockerimgfull,...
                          'gpuid',fallback_gpuid,...
                          'apiver',apt.docker_api_version(), ...
                          varargin{:}) ;  % key-value pairs in varagin will override ones specified here
    
      % Wrap for ssh'ing into a remote docker host, if needed
      if ~isempty(obj.dockerremotehost),
        codestr = wrapCommandSSH(codestr,'host',obj.dockerremotehost);
      end
    end  % function 
    
    function [didLaunchSucceed, instanceID] = launchNewAWSInstance(obj)
      if obj.type ~= DLBackEnd.AWS 
        error('Can only launch new AWS EC2 instance when backend is AWS') ;
      end
      [didLaunchSucceed, instanceID] = obj.awsec2.launchNewInstance() ;
    end
    
    function result = get.condaEnv(obj)
      if obj.didOverrideDefaultCondaEnv_
        result = obj.customCondaEnv_ ;
      else
        result = DLBackEndClass.default_conda_env ;
      end      
    end  % function

    function set.condaEnv(obj, newValue)
      if ischar(newValue) && ~isempty(newValue) && isrow(newValue)
        % all is well
      else
        error('APT:invalidValue', '"%s" is a not valid value for the conda environment', newValue);
      end
      obj.customCondaEnv_ = newValue ;
      obj.didOverrideDefaultCondaEnv_ = true ;
    end  % function
    
    function result = get.dockerimgroot(obj)
      if obj.didOverrideDefaultDockerImgSpec_
        result = obj.customDockerImgRoot_ ;
      else
        result = DLBackEndClass.defaultDockerImgRoot ;
      end      
    end

    function set.dockerimgroot(obj, newValue)
      if ischar(newValue) && ~isempty(newValue) && isrow(newValue)
        % all is well
      else
        error('APT:invalidValue', '"%s" is a not valid value for the Docker image root', newValue);
      end
      obj.customDockerImgRoot_ = newValue ;
      % There's only a single didOverride field for the docker image spec, so need
      % to make sure .customDockerImgTag_ gets set if needed.
      if obj.didOverrideDefaultDockerImgSpec_
        % nothing else to do
      else
        obj.didOverrideDefaultDockerImgSpec_ = true ;
        obj.customDockerImgTag_ = DLBackEndClass.defaultDockerImgTag ;
      end
    end

    function result = get.dockerimgtag(obj)
      if obj.didOverrideDefaultDockerImgSpec_
        result = obj.customDockerImgTag_ ;
      else
        result = DLBackEndClass.defaultDockerImgTag ;
      end      
    end
    
    function set.dockerimgtag(obj, newValue)
      if ischar(newValue) && ~isempty(newValue) && isrow(newValue)
        % all is well
      else
        error('APT:invalidValue', '"%s" is a not valid value for the Docker image tag', newValue);
      end
      obj.customDockerImgTag_ = newValue ;
      % There's only a single didOverride field for the docker image spec, so need
      % to make sure .customDockerImgRoot_ gets set if needed.
      if obj.didOverrideDefaultDockerImgSpec_
        % nothing else to do
      else
        obj.didOverrideDefaultDockerImgSpec_ = true ;
        obj.customDockerImgRoot_ = DLBackEndClass.defaultDockerImgRoot ;
      end
    end

    function testBackendConfig(obj, labeler)
      obj.testText_ = {''};
      labeler.notify('updateBackendTestText') ;
      switch obj.type,
        case DLBackEnd.Bsub,
          obj.testBsubBackendConfig_(labeler) ;
        case DLBackEnd.Docker
          obj.testDockerBackendConfig_(labeler) ;
        case DLBackEnd.AWS
          obj.awsec2.testBackendConfig(obj, labeler) ;
        case DLBackEnd.Conda
          obj.testCondaBackendConfig_(labeler) ;
        otherwise
          error('Tests for %s have not been implemented', obj.type) ;
      end      
    end  % function

    function text = testText(obj)
      % Get test text
      text = obj.testText_;
    end  % function

    function result = changeToTrackInfoPathsToRemoteFromWsl_(obj, totrackinfo)
      % Convert all paths in totrackinfo, which should be wsl paths encoded as char
      % arrays, to their corresponding remote paths on the backend.  This method
      % does not mutate obj or the input totrackinfo.  result is similar to
      % totrckinfo but with wsl paths replaced with remote.  If backend is non-aws,
      % this is essentially the identity function, but the returned (handle) object
      % is a copy of the input.

      % If backend has local filesystem (i.e. is not AWS), this function is the identity function
      if ~isequal(obj.type,DLBackEnd.AWS)
        result = totrackinfo.copy() ;
        return
      end
      
      % Generate all the relocated paths
      result = obj.awsec2.changeToTrackInfoPathsToRemoteFromWsl(totrackinfo) ;
    end  % function    

    function result = convertWslShellCommandToRemote(obj, inputCommand)
      if isequal(obj.type, DLBackEnd.AWS)
        result = obj.awsec2.convertWslShellCommandToRemote(inputCommand) ;
      else
        result = DLBackEndClass.convertWslShellCommandToRemoteForNonAwsStatic_(inputCommand) ;
      end
    end  % end function
  end  % methods

  methods (Static)
    function result = repoSnapshotCmd(aptRootRemotePath, dotAptSnapshotRemotePath)
      repoSnapshotScriptRemotePath = aptRootRemotePath.cat('matlab', 'repo_snapshot.sh') ;
      result = apt.ShellCommand({repoSnapshotScriptRemotePath, aptRootRemotePath, '>', dotAptSnapshotRemotePath}, ...
                                apt.PathLocale.remote, ...
                                apt.Platform.posix) ;
    end
  end  % methods (Static)

  methods (Static)
    function result = convertWslShellCommandToRemoteForNonAwsStatic_(inputCommand)
      % Convert command to remote locale by converting all path tokens, for non-AWS
      % backends.
      %
      % Returns:
      %   apt.ShellCommand: New command with paths converted to remote locale.

      function result = processToken(token)
        % Local function to convert each token to target locale
        if isa(token, 'apt.MetaPath')
          result = DLBackEndClass.convertWslMetaPathToRemoteForNonAwsStatic_(token);
        elseif isa(token, 'apt.ShellCommand')
          result = DLBackEndClass.convertWslShellCommandToRemoteForNonAwsStatic_(token);
        elseif isa(token, 'apt.ShellBind')
          originalSourcePath = token.sourcePath ;
          originalDestPath = token.destPath ;
          newSourcePath =  DLBackEndClass.convertWslMetaPathToRemoteForNonAwsStatic_(originalSourcePath) ;
          newDestPath =  DLBackEndClass.convertWslMetaPathToRemoteForNonAwsStatic_(originalDestPath) ;
          result = apt.ShellBind(newSourcePath, newDestPath) ;
        elseif isa(token, 'apt.ShellVariableAssignment')
          originalValue = token.value ;
          if isa(originalValue, 'apt.MetaPath')
            newValue = DLBackEndClass.convertWslMetaPathToRemoteForNonAwsStatic_(originalValue) ;
            result = apt.ShellVariableAssignment(token.identifier, newValue) ;            
          else
            result = token ;
          end
        elseif isa(token, 'apt.ShellLiteral')
          result = token;
        else
          error('Internal error: Unhandled ShellToken subclass in DLBackEndClass.convertWslShellCommandToRemoteForNonAwsStatic_()') ;
        end
      end  % local function

      % Use cellfun to process all tokens
      newTokens = cellfun(@processToken, inputCommand.tokens, 'UniformOutput', false);

      result = apt.ShellCommand(newTokens, apt.PathLocale.remote, inputCommand.platform);
    end

    function result = convertWslMetaPathToRemoteForNonAwsStatic_(inputWslMetaPath)
      % Convert WSL MetaPath to remote by replacing prefix based on file role
      %
      % Args:
      %   inputWslMetaPath (apt.MetaPath): WSL path to convert
      %
      % Returns:
      %   apt.MetaPath: MetaPath with WSL prefix replaced by remote equivalent.
      %   (Which for non-AWS backends is just the same.)
      
      assert(isa(inputWslMetaPath, 'apt.MetaPath'), 'wslMetaPath must be an apt.MetaPath');
      assert(inputWslMetaPath.locale == apt.PathLocale.wsl, 'wslMetaPath must have WSL locale');

      % Just return a MetaPath with the locale set to remote
      path = inputWslMetaPath.path ;
      role = inputWslMetaPath.role ;
      result = apt.MetaPath(path, apt.PathLocale.remote, role) ;
    end  % function
  end  % methods (Static)  
end  % classdef

