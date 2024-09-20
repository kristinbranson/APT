classdef DLBackEndClass < matlab.mixin.Copyable
  % 
  % APT Backends specify a physical machine/server where GPU code is run.
  %
  % * DLNetType: what DL net is run
  % * DLBackEndClass: where/how DL was run
  %
  % TODO: Codegen methods should be moved out of DeepTracker and into this class
  % and use instance state (eg, docker codegen for current tag; bsub for
  % specified .sif; etc). Conceptually this class would just be "DLBackEnd" and
  % the enum/type would go away.
  
  properties (Constant)
    minFreeMem = 9000  % in MiB
    defaultDockerImgTag = 'apt_20230427_tf211_pytorch113_ampere'
    defaultDockerImgRoot = 'bransonlabapt/apt_docker'
 
    RemoteAWSCacheDir = '/home/ubuntu/cacheDL'

    jrchost = 'login1.int.janelia.org'
    jrcprefix = ''
    jrcprodrepo = '/groups/branson/bransonlab/apt/repo/prod'

    default_conda_env = 'APT'
    default_singularity_image_path = '/groups/branson/bransonlab/apt/sif/apt_20230427_tf211_pytorch113_ampere.sif' 
    legacy_default_singularity_image_path = '/groups/branson/bransonlab/apt/sif/prod.sif'
    legacy_default_singularity_image_path_for_detect = '/groups/branson/bransonlab/apt/sif/det.sif'

    default_docker_api_version = '1.40'
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
    dockerapiver = DLBackEndClass.default_docker_api_version  % docker codegen will occur against this docker api ver
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

    condaEnv = DLBackEndClass.default_conda_env   % used only for Conda

    % We set these to the string 'invalid' so we can catch them in loadobj()
    % They are set properly in the constructor.
    singularity_image_path_ = '<invalid>'
    does_have_special_singularity_detection_image_path_ = '<invalid>'
    singularity_detection_image_path_ = '<invalid>'

    % Used to keep track of whether movies have been uploaded or not
    didUploadMovies_ = false

    % When we upload movies, keep track of the correspondence, so we can help the
    % consumer map between the paths
    localPathFromMovieIndex_ = cell(1,0) ;
    remotePathFromMovieIndex_ = cell(1,0) ;

    % The job registry.  These are protected, transient in spirit.
    % These are jobs that can be spawned with a subsequent call to
    % spawnRegisteredJobs().
    syscmds_ = cell(0,1)
    cmdfiles_ = cell(0,1)
    logcmds_ = cell(0,1)
  end

  properties (Dependent)
    filesep
    dockerimgfull % full docker img spec (with tag if specified)
    singularity_image_path
    singularity_detection_image_path
    isInAwsDebugMode
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
      % Just populate this now, whether or not we end up using it      
      obj.awsec2 = AWSec2() ;
    end
  end
  
  methods % Prop access
    function v = get.filesep(obj)
      if obj.type == DLBackEnd.Conda,
        v = filesep();  %#ok<CPROP> 
      else
        v = '/';
      end
    end

    function set.type(obj, value)
      assert(isa(value, 'DLBackEnd')) ;
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
    function [return_code, stdouterr] = runFilesystemCommand(obj, basecmd, varargin)
      % Run the basecmd using apt.syscmd(), after wrapping suitably for the type of
      % backend.  Unlike spawn(), this blocks, and doesn't return a process
      % identifier of any kind.  Return values are like those from system(): a
      % numeric return code and a string containing any command output.
      % Note that any file names in the basecmd must refer to the filenames on the
      % *backend* filesystem.
      %failbehavior = myparse(varargin, 'failbehavior', 'silent') ;  % want different default than apt.syscmd() provides 
      %leftover_args = remove_pair_from_key_value_list_if_present(varargin, 'failbehavior') ;
      switch obj.type,
        case DLBackEnd.AWS
          command = wrapFilesystemCommandForAWSBackend(basecmd, obj) ;
        case DLBackEnd.Bsub,
          % For now, we assume Matlab frontend is running on a JRC cluster node,
          % which means the filesystem is local.
          command = basecmd ;
        case DLBackEnd.Conda
          command = basecmd ;
        case DLBackEnd.Docker
          % Don't support remote docker host anymore.
          command = basecmd ;
        otherwise
          error('Not implemented: %s',obj.type);
      end
      [return_code, stdouterr] = apt.syscmd(command, 'failbehavior', 'silent', 'verbose', false, varargin{:}) ;
        % Things passed in with varargin should overide things we set here
    end  % function

    function jobID = parseJobID(obj, res)
      switch obj.type
        case DLBackEnd.AWS,
          jobID = DLBackEndClass.parseJobIdAws(res) ;
        case DLBackEnd.Bsub,
          jobID = DLBackEndClass.parseJobIDBsub(res);
        case DLBackEnd.Conda,
          jobID = DLBackEndClass.parseJobIDConda(res);
        case DLBackEnd.Docker,
          jobID = DLBackEndClass.parseJobIDDocker(res);
        otherwise
          error('Not implemented: %s',obj.type);
      end
    end  % function

    function result = remoteMoviePathFromLocal(obj, localPath)
      % Convert a local movie path to the remote equivalent.
      % For non-AWS backends, this is the identity function.
      if isequal(obj.type, DLBackEnd.AWS) ,
        movieName = fileparts23(localPath) ;
        remoteMovieFolderPath = linux_fullfile(DLBackEndClass.RemoteAWSCacheDir, 'movies') ;
        rawRemotePath = linux_fullfile(remoteMovieFolderPath, movieName) ;
        result = FSPath.standardPath(rawRemotePath);  % transform to standardized linux-style path
      else
        result = localPath ;
      end
    end

    function result = remoteMoviePathsFromLocal(obj, localPathFromMovieIndex)
      % Convert a cell array of local movie paths to their remote equivalents.
      % For non-AWS backends, this is the identity function.
      result = cellfun(@(path)(obj.remoteMoviePathFromLocal(path)), localPathFromMovieIndex, 'UniformOutput', false) ;
    end

    function uploadMovies(obj, localPathFromMovieIndex)
      % Upload movies to the backend, if necessary.
      if ~isequal(obj.type, DLBackEnd.AWS) ,
        obj.didUploadMovies_ = true ;
        return
      end
      if obj.didUploadMovies_ ,
        return
      end
      remotePathFromMovieIndex = obj.remoteMoviePathsFromLocal(localPathFromMovieIndex) ;
      movieCount = numel(localPathFromMovieIndex) ;
      fprintf('Uploading %d movie files...\n', movieCount) ;
      fileDescription = 'Movie file' ;
      sidecarDescription = 'Movie sidecar file' ;
      for i = 1:movieCount ,
        localPath = localPathFromMovieIndex{i};
        remotePath = remotePathFromMovieIndex{i};
        obj.uploadOrVerifySingleFile_(localPath, remotePath, fileDescription) ;  % throws
        % If there's a sidecar file, upload it too
        [~,~,fileExtension] = fileparts(localPath) ;
        if strcmp(fileExtension,'.mjpg') ,
          sidecarLocalPath = FSPath.replaceExtension(localPath, '.txt') ;
          if exist(sidecarLocalPath, 'file') ,
            sidecarRemotePath = obj.remoteMoviePathFromLocal(sidecarLocalPath) ;
            obj.uploadOrVerifySingleFile_(sidecarLocalPath, sidecarRemotePath, sidecarDescription) ;  % throws
          end
        end
      end      
      fprintf('Done uploading %d movie files.\n', movieCount) ;
      obj.didUploadMovies_ = true ; 
      obj.localPathFromMovieIndex_ = localPathFromMovieIndex ;
      obj.remotePathFromMovieIndex_ = remotePathFromMovieIndex ;
    end  % function

    function uploadOrVerifySingleFile_(obj, localPath, remotePath, fileDescription)
      % Upload a single file.  Protected by convention.
      % Doesn't check to see if the backend type has a different filesystem.  That's
      % why outsiders shouldn't call it.
      localFileDirOutput = dir(localPath) ;
      localFileSizeInKibibytes = round(localFileDirOutput.bytes/2^10) ;
      % We just use scpUploadOrVerify which does not confirm the identity
      % of file if it already exists. These movie files should be
      % immutable once created and their naming (underneath timestamped
      % modelchainIDs etc) should be pretty/totally unique. 
      %
      % Only situation that might cause problems are augmentedtrains but
      % let's not worry about that for now.
      localFileName = localFileDirOutput.name ;
      fullFileDescription = sprintf('%s (%s), %d KiB', fileDescription, localFileName, localFileSizeInKibibytes) ;
      obj.scpUploadOrVerify(localPath, ...
                            remotePath, ...
                            fullFileDescription, ...
                            'destRelative',false) ;  % throws      
    end

    function delete(obj)  %#ok<INUSD> 
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
      obj.stopEc2InstanceIfNeeded_() ;
    end
    
    function obj2 = copyAndDetach(obj)
      % See notes in BgClient, BgWorkerObjAWS.
      %
      % Sometimes we want a deep-copy of obj that is sanitized for
      % eg serialization. This copy may still be largely functional (in the
      % case of BgWorkerObjAWS) or perhaps it can be 'reconstituted' at
      % load-time as here.
      
      assert(isscalar(obj));
      obj2 = copy(obj);
%       if ~isempty(obj2.awsec2)
%         obj2.awsec2.clearStatusFuns();
%       end
    end  % function
  end  % methods

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
      % 20220728 Win/Conda migration to WSL2/Docker
      if obj.type==DLBackEnd.Conda
        if ispc() ,
          warningNoTrace(...
            ['Updating Windows backend from Conda -> Docker. ' ...
             'If you have not already, please see the documentation for Windows/WSL2 setup instructions.']);
          obj.type = DLBackEnd.Docker;
        else
          warningNoTrace('Current backend is Conda.  This is only intended for developers.  Be careful.');
        end
      end
%       if obj.type==DLBackEnd.Docker || obj.type==DLBackEnd.Bsub
%         defaultDockerFullImageSpec = ...
%           horzcat(DLBackEndClass.defaultDockerImgRoot, ':', DLBackEndClass.defaultDockerImgTag) ;
%         fullImageSpec = obj.dockerimgfull ;
%         if ~strcmp(fullImageSpec, defaultDockerFullImageSpec) ,
%           message = ...
%             sprintf('The Docker image spec (%s) differs from the default (%s).', ...
%                     fullImageSpec, ...
%                     defaultDockerFullImageSpec) ;
%           warningNoTrace(message) ;
%         end
%       end
      % 20211101 turn on by default
      obj.jrcsimplebindpaths = 1;
      % In modern versions, we always have a .awsec2, whether we need it or not
      if isempty(obj.awsec2) ,
        obj.awsec2 = AWSec2() ;
      end
      % On load, clear the .awsec2 fields that should be Transient, but can't be b/c
      % we need them to survive going through parfeval()
      obj.awsec2.instanceIP = [] ;
      obj.awsec2.remotePID = [] ;
      obj.awsec2.isInDebugMode = false ;
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
        didLaunch = false;
        if ~obj.awsec2.isConfigured || ~obj.awsec2.isSpecified,
          [tfsucc,instanceID,~,reason,didLaunch] = ...
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
      elseif obj.type==DLBackEnd.Conda ,
        if ispc() ,
          tf = false ;
          reason = 'Conda backend is not supported on Windows.' ;
        else
          tf = true ;
          reason = '' ;
        end
      else
        tf = true;
        reason = '';
      end
    end  % method
    
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

    function v = isGpuLocal(obj)
      % Docker and Conda backends use a local GPU, Bsub (i.e. Janelia LSF) and AWS
      % do not.  Basically this refers to whether the Python training/tracking code
      % will run on the same machine as the Matlab frontend.
      v = isequal(obj.type, DLBackEnd.Docker) || isequal(obj.type,DLBackEnd.Conda) ;
    end
    
    function v = isGpuRemote(obj)
      v = ~obj.isGpuLocal(obj) ;
    end
    
    function v = isFilesystemLocal(obj)
      % The conda and bsub (i.e. Janelia LSF) and Docker backends share (mostly) the
      % same filesystem as the Matlab process.  AWS does not.
      v = isequal(obj.type,DLBackEnd.Conda) || isequal(obj.type,DLBackEnd.Bsub) || isequal(obj.type,DLBackEnd.Docker) ;
    end
    
%     function v = isFilesystemRemote(obj)
%       v = ~obj.isFilesystemLocal(obj) ;
%     end
    
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

%       global FORCEONEJOB;
%       if isequal(FORCEONEJOB,true),
%         warning('Forcing one GPU job');
%         ngpu = 1;
%       end
      
      freemem = freemem(1:ngpu);
      gpuid = gpuid(1:ngpu);
      
      ngpureturn = min(ngpu,nrequest);
      gpuid = gpuid(1:ngpureturn);

      freemem = freemem(1:ngpureturn);
      
      obj.gpuids = gpuid;
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
        
    function tfSucc = writeCmdToFile(obj, syscmds, cmdfiles, jobdesc)
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
%         [fh,msg] = fopen(cmdfiles{i},'w');
%         if isequal(fh,-1)
%           warningNoTrace('Could not open command file ''%s'': %s',cmdfile,msg);
%         else
%           fprintf(fh,'%s\n',syscmds{i});
%           fclose(fh);
%           fprintf(1,'Wrote command for %s %d to cmdfile %s.\n',jobdesc,i,cmdfiles{i});
%           tfSucc(i) = true;
%         end
      end  % for
    end  % function
  end  % methods block
  
  methods (Static)
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
        warning('Could not parse job id from:\n%s\n',res);
      end
    end

    function jobID = parseJobIdAws(res)
      %fprintf('res: %s', res) ;
      jobID = DLBackEndClass.parseJobIDDocker(res) ;
%       jobid = str2double(strtrim(res)) ;
%       if isnan(jobid) 
%         warning('Could not parse job id from:\n%s\n',res);
%       end
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
        warning('Could not parse job id from:\n%s\n',res);
        disp(getReport(ME));
        jobID = '';
      end
    end

    function jobid = parseJobIDConda(res)
      % We look for the first line of res that is an integer. If we can't find such
      % a line, we throw a warning and return nan.

      %fprintf('res: %s', res) ;      
      line_from_line_index = break_string_into_lines(res) ;
      line_count = numel(line_from_line_index) ;
      for line_index = 1 : line_count ,
        line = line_from_line_index{line_index} ;         
        jobid = str2double(strtrim(line)) ;
        if isfinite(jobid) && round(jobid)==jobid ,
          return
        end
      end
      % If get here, we have failed to read a job id
      warning('Could not parse job id from:\n%s\n',res);
      jobid = nan ;
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
      
      [~,hedit] = DLBackEndClass.createFigTestConfig('Test JRC Cluster Backend');
      hedit.String = {sprintf('%s: Testing JRC cluster backend...',datestr(now))};
      drawnow;
      
      % test that you can ping jrc host
      hedit.String{end+1} = ''; drawnow;
      hedit.String{end+1} = sprintf('** Testing that host %s can be reached...\n',host); drawnow;
      cmd = sprintf('ping -c 1 -W 10 %s',host);
      hedit.String{end+1} = cmd; drawnow;
      [status,result] = apt.syscmd(cmd);
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
      cmd1 = wrapCommandSSH(remotecmd,'host',host,'timeout',timeout);
      %cmd = sprintf('timeout 20 %s',cmd1);
      cmd = cmd1;
      hedit.String{end+1} = cmd; drawnow;
      [status,result] = apt.syscmd(cmd);
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
      cmd = wrapCommandSSH(remotecmd,'host',host);
      hedit.String{end+1} = cmd; drawnow;
      [status,result] = apt.syscmd(cmd);
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

    function aptroot = bsubSetRootUpdateRepo_(obj)
      aptroot = apt.bsubSetRootUpdateRepo(obj.deepnetrunlocal) ;
      obj.bsubaptroot = aptroot;
    end

%     function bsubPretrack(obj,cacheDir)
%       obj.bsubSetRootUpdateRepo_(cacheDir);
%     end
    
  end
  
  methods % Docker

%     function [cmd,cmdend] = dockercmd(obj)
%       apiVerExport = sprintf('export DOCKER_API_VERSION=%s;',obj.dockerapiver);
%       remHost = obj.dockerremotehost;
%       if isempty(remHost),
%         cmd = sprintf('%s docker',apiVerExport);
%         cmdend = '';
%       else
%         cmd = sprintf('ssh -t %s "%s docker',remHost,apiVerExport);
%         cmdend = '"';        
%       end
%       if ispc
%         cmd = ['wsl ' cmd];
%       end
%     end

    function s = dockercmd(obj)
      s = dockercmd_from_apiver(obj.dockerapiver) ;
    end

    % KB 20191219: moved this to not be a static function so that we could
    % use this object's dockerremotehost
    function [tfsucc,clientver,clientapiver] = getDockerVers(obj)
      % Run docker cli to get docker versions
      %
      % tfsucc: true if docker cli command successful
      % clientver: if tfsucc, char containing client version; indeterminate otherwise
      % clientapiver: if tfsucc, char containing client apiversion; indeterminate otherwise
      
      dockercmd = obj.dockercmd();      
      FMTSPEC = '{{.Client.Version}}#{{.Client.DefaultAPIVersion}}';
      cmd = sprintf('%s version --format "%s"',dockercmd,FMTSPEC);
%       if ~isempty(obj.dockerremotehost),
%         cmd = wrapCommandSSH(cmd,'host',obj.dockerremotehost);
%       end
%      cmd = wrap_linux_command_line_for_wsl_if_windows(cmd) ;
      
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
    
    function [tfsucc,hedit] = testDockerConfig(obj)
      tfsucc = false;

      [~,hedit] = DLBackEndClass.createFigTestConfig('Test Docker Configuration');      
      hedit.String = {sprintf('%s: Testing Docker Configuration...',datestr(now))}; 
      drawnow;
      
      % docker hello world
      hedit.String{end+1} = ''; drawnow;
      hedit.String{end+1} = '** Testing docker hello-world...'; drawnow;
      
      dockercmd = obj.dockercmd();
      cmd = sprintf('%s run --rm hello-world',dockercmd);

%       if ~isempty(obj.dockerremotehost),
%         cmd = wrapCommandSSH(cmd,'host',obj.dockerremotehost);
%       end

%       cmd = wrap_linux_command_line_for_wsl_if_windows(cmd) ;
      
      fprintf(1,'%s\n',cmd);
      hedit.String{end+1} = cmd; 
      drawnow;
      [st,res] = apt.syscmd(cmd);
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
      
      [tfsucc,~,clientapiver] = obj.getDockerVers();
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
      cmd = wrapCommandDocker(basecmd,...
                              'dockerimg', obj.dockerimgfull, ...
                              'containername','containerTest',...
                              'detach',false,...
                              'bindpath',{deepnetroot,homedir});
      hedit.String{end+1} = cmd;
      RUNAPTHELLO = 1;
      if RUNAPTHELLO % AL: this may not work property on a multi-GPU machine with some GPUs in use
        %fprintf(1,'%s\n',cmd);
        %hedit.String{end+1} = cmd; drawnow;
        [st,res] = apt.syscmd(cmd);
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
      [gpuid,~,~] = obj.getFreeGPUs(1,'verbose',true);
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

      [~,hedit] = DLBackEndClass.createFigTestConfig('Test Conda Configuration');      
      hedit.String = {sprintf('%s: Testing Conda Configuration...',datestr(now))}; 
      drawnow;

      % Check if Windows box.  Conda backend is not supported on Windows.
      hedit.String{end+1} = ''; drawnow;
      hedit.String{end+1} = '** Checking for (lack of) Windows...'; drawnow;
      if ispc() ,
        hedit.String{end+1} = 'FAILURE. Conda backend is not supported on Windows.'; drawnow;
        return
      end
      hedit.String{end+1} = 'SUCCESS!'; drawnow;

      % make sure conda is installed
      hedit.String{end+1} = ''; drawnow;
      hedit.String{end+1} = '** Checking for conda...'; drawnow;
      conda_executable_path = find_conda_executable() ;
      cmd = sprintf('%s -V', conda_executable_path) ;
      hedit.String{end+1} = cmd; drawnow;
      [st,~] = apt.syscmd(cmd);
      %reslines = splitlines(res);
      if st~=0
        hedit.String{end+1} = sprintf('FAILURE. Error with ''%s''. Make sure you have installed conda and added it to your PATH.',cmd); drawnow;
        return;
      end
      hedit.String{end+1} = 'SUCCESS!'; drawnow;


      % activate APT
      hedit.String{end+1} = ''; drawnow;
      hedit.String{end+1} = sprintf('** Testing conda run -n %s...', obj.condaEnv); 
      drawnow;

      raw_cmd = 'echo "Hello, world!"' ;
      cmd = wrapCommandConda(raw_cmd, 'condaEnv', obj.condaEnv) ;
      %fprintf(1,'%s\n',cmd);
      hedit.String{end+1} = cmd; drawnow;
      [st,~] = apt.syscmd(cmd);
      %reslines = splitlines(res);
      %reslinesdisp = reslines(1:min(4,end));
      %hedit.String = [hedit.String; reslinesdisp(:)];
      if st~=0
        hedit.String{end+1} = sprintf('FAILURE. Error with ''%s''. Make sure you have created the conda environment %s',cmd, obj.condaEnv); 
        drawnow;
        return
      end
      hedit.String{end+1} = 'SUCCESS!'; drawnow;
        
      % free GPUs
      hedit.String{end+1} = ''; drawnow;
      hedit.String{end+1} = '** Looking for free GPUs ...'; drawnow;
      gpuid = obj.getFreeGPUs(1,'verbose',true);
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
    
    function [tfsucc,hedit] = testAWSConfig(obj,varargin)  %#ok<INUSD> 
      tfsucc = false;
      [~,hedit] = DLBackEndClass.createFigTestConfig('Test AWS Backend');
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
        [status,result] = apt.syscmd(cmd);
        hedit.String{end+1} = result; drawnow;
        if status ~= 0,
          hedit.String{end+1} = 'FAILURE. Did not find ssh.'; drawnow;
          return;
        end
      end
      
      if ispc(),
        hedit.String{end+1} = sprintf('\n** Testing that certUtil is installed...\n'); drawnow;
        cmd = 'where certUtil';
        hedit.String{end+1} = cmd; drawnow;
        [status,result] = apt.syscmd(cmd);
        hedit.String{end+1} = result; drawnow;
        if status ~= 0,
          hedit.String{end+1} = 'FAILURE. Did not find certUtil.'; drawnow;
          return;
        end
      end
      
      % test that AWS CLI is installed
      hedit.String{end+1} = sprintf('\n** Testing that AWS CLI is installed...\n'); drawnow;
      cmd = 'aws ec2 describe-regions --output table';
      hedit.String{end+1} = cmd; drawnow;
      [status,result] = AWSec2.syscmd(cmd);
      tfsucc = (status==0) ;      
      hedit.String{end+1} = result; drawnow;
      if ~tfsucc % status ~= 0,
        hedit.String{end+1} = 'FAILURE. Error using the AWS CLI.'; drawnow;
        return
      end

      % test that apt_dl security group has been created
      hedit.String{end+1} = sprintf('\n** Testing that apt_dl security group has been created...\n'); drawnow;
      cmd = 'aws ec2 describe-security-groups';
      hedit.String{end+1} = cmd; drawnow;
      [status,result] = AWSec2.syscmd(cmd,'isjsonout',true);
      tfsucc = (status==0) ;
      if status == 0,
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
        return
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
%       cmd1 = wrapCommandSSH(remotecmd,'host',host);
%       cmd = sprintf('timeout 20 %s',cmd1);
%       hedit.String{end+1} = cmd; drawnow;
%       [status,result] = apt.syscmd(cmd);
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
%       cmd = wrapCommandSSH(remotecmd,'host',host);
%       hedit.String{end+1} = cmd; drawnow;
%       [status,result] = apt.syscmd(cmd);
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
    
    function checkConnection(obj)  
      % Errors if connection to backend is ok.  Otherwise returns nothing.
      if isequal(obj.type, DLBackEnd.AWS) ,
        aws = obj.awsec2;
        aws.checkInstanceRunning() ;
      end
    end

    function scpUploadOrVerify(obj, varargin)
      if isequal(obj.type, DLBackEnd.AWS) ,
        aws = obj.awsec2;
        aws.scpUploadOrVerify(varargin{:}) ;
      end      
    end

    function rsyncUpload(obj, src, dest)
      if isequal(obj.type, DLBackEnd.AWS) ,
        aws = obj.awsec2 ;
        aws.rsyncUpload(src, dest) ;
      end      
    end
    
    function aptroot = awsUpdateRepo_(obj)  % throws if fails      
      % What branch do we want?
      branch = 'dockerized-aws' ;  % TODO: For debugging only, set back to develop or main eventually

      % Clone the git repo if needed
      obj.cloneAWSRemoteAPTRepoIfNeeeded_(branch) ;

      % Determine the remote APT source root, and quote it for bash
      aptroot = obj.getAPTRoot() ;
      quoted_aptroot = escape_string_for_bash(aptroot) ;

      % Checkout the correct branch
      command_line_1 = sprintf('git -C %s checkout %s', quoted_aptroot, branch) ;
      [st_1,res_1] = obj.runFilesystemCommand(command_line_1) ;
      if st_1 ~= 0 ,
        error('Failed to update remote APT repo:\n%s', res_1);
      end

      % Do a git pull
      command_line_2 = sprintf('git -C %s pull', quoted_aptroot) ;
      [st_2,res_2] = obj.runFilesystemCommand(command_line_2) ;
      if st_2 ~= 0 ,
        error('Failed to update remote APT repo:\n%s', res_2);
      end
      
      % Run the remote Python script to download the pretrained model weights
      % This python script doesn't do anything fancy, apparently, so we use the
      % python interpreter provided by the plain EC2 instance, not the one inside
      % the Docker container on the instance.
      download_script_path = linux_fullfile(aptroot, 'deepnet', 'download_pretrained.py') ;
      quoted_download_script_path = escape_string_for_bash(download_script_path) ;      
      [st_3,res_3] = obj.runFilesystemCommand(quoted_download_script_path) ;
      if st_3 ~= 0 ,
        error('Failed to update remote APT repo:\n%s', res_3);
      end
      
      % If get here, all is well
      fprintf('Updated remote APT repo.\n\n');
    end  % function
    
    function cloneAWSRemoteAPTRepoIfNeeeded_(obj, branch_name)
      % Clone the APT source repo on the remote AWS instance, if needed.
      
      % Does the APT root dir exist?
      remote_apt_root = obj.getAPTRoot() ;
      does_remote_apt_dir_exist = obj.exist(remote_apt_root) ;
      
      % clone it if needed
      if does_remote_apt_dir_exist ,
        fprintf('Found JRC/APT repo at %s.\n', remote_apt_root);
      else
        apt_github_url = 'https://github.com/kristinbranson/APT' ;
        command_line = sprintf('git clone -b %s %s %s', branch_name, apt_github_url, remote_apt_root);
        [return_code, stdouterr] = obj.runFilesystemCommand(command_line) ;
        if return_code ~= 0 ,
          error('Unable to clone APT git repo in AWS instance.\nReturn code: %d\nStdout/stderr:\n%s\n', return_code, stdouterr) ;
        end
        fprintf('Cloned APT repo into %s on AWS instance.\n', remote_apt_root);
      end  % if
    end  % function    
  end  % public methods block

  methods
    function [didsucceed, msg] = mkdir(obj, dir_name)
      % Create the named directory, either locally or remotely, depending on the
      % backend type.
      quoted_dirloc = escape_string_for_bash(dir_name) ;
      base_command = sprintf('mkdir -p %s', quoted_dirloc) ;
      [status, msg] = obj.runFilesystemCommand(base_command) ;
      didsucceed = (status==0) ;
    end

    function [didsucceed, msg] = deleteFile(obj, file_name)
      % Delete the named file, either locally or remotely, depending on the
      % backend type.
      quoted_file_name = escape_string_for_bash(file_name) ;
      base_command = sprintf('rm %s', quoted_file_name) ;
      [status, msg] = obj.runFilesystemCommand(base_command) ;
      didsucceed = (status==0) ;
    end

    function [doesexist, msg] = exist(obj, file_name, file_type)
      % Check whether the named file/dir exists, either locally or remotely,
      % depending on the backend type.
      if ~exist('file_type', 'var') ,
        file_type = '' ;
      end
      if strcmpi(file_type, 'dir') ,
        option = '-d' ;
      elseif strcmpi(file_type, 'file') ,
        option = '-f' ;
      else
        option = '-e' ;
      end
      quoted_file_name = escape_string_for_bash(file_name) ;
      base_command = sprintf('test %s %s', option, quoted_file_name) ;
      [status, msg] = obj.runFilesystemCommand(base_command) ;
      doesexist = (status==0) ;
    end

    function [didSucceed, errorMessage] = writeStringToFile(obj, filename, str)
      % Write the given string to a file, overrwriting any previous contents.
      % For remote backends, uses a single "ssh echo $string > $filename" to do
      % this, so limited to strings of ~10^5 bytes.
      if obj.isFilesystemLocal() ,
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
        [status, msg] = obj.runFilesystemCommand(base_command) ;
        if status ~= 0 ,
          didSucceed = false ;
          errorMessage = sprintf('Something went wrong while writing to backend file %s: %s',filename,msg);
          return
        end
      end
      didSucceed = true ;
      errorMessage = '' ;
    end  % function    

    function aptroot = updateRepo(obj, localCacheDir)  %#ok<INUSD> 
      % Update the APT repo on the backend.  While we're at it, make sure the
      % pretrained weights are downloaded.  The method formerly known as
      % setupForTrainingOrTracking().
      % localCacheDir should be e.g. /home/joeuser/.apt/tp662830c8_246a_49c6_816c_470db4ecd950
      % localCacheDir is not currently used, but will be needed to get the JRC
      % backend working properly for AD-linked Linux workstations.
      switch obj.type
        case DLBackEnd.Bsub ,
          aptroot = obj.bsubSetRootUpdateRepo_();
        case {DLBackEnd.Conda, DLBackEnd.Docker} ,
          aptroot = APT.Root;
          apt.downloadPretrainedWeights('aptroot', aptroot) ;
        case DLBackEnd.AWS ,
          obj.awsec2.checkInstanceRunning();  % errs if instance isn't running
          aptroot = obj.awsUpdateRepo_();  % this is the remote APT root
        otherwise
          error('Unknown backend type') ;
      end
    end  % function    

    function result = getLocalMoviePathFromRemote(obj, queryRemotePath)
      if ~obj.didUploadMovies_ ,
        error('Can''t get a local movie path from a remote path if movies have not been uploaded.') ;
      end
      movieCount = numel(obj.remotePathFromMovieIndex_) ;
      for movieIndex = 1 : movieCount ,
        remotePath = obj.remotePathFromMovieIndex_{movieIndex} ;
        if strcmp(remotePath, queryRemotePath) ,
          result = obj.localPathFromMovieIndex_{movieIndex} ;
          return
        end
      end
      % If we get here, queryRemotePath did not match any path in obj.remotePathFromMovieIndex_
      error('Query path %s does not match any remote movie path known to the backend.', queryRemotePath) ;
    end  % function
  end  % public methods block

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
      obj = larva ;
      if strcmp(larva.singularity_image_path_, '<invalid>') ,
        % This must come from an older .mat file, so we use the legacy values
        obj.singularity_image_path_ = DLBackEndClass.legacy_default_singularity_image_path ;
        obj.does_have_special_singularity_detection_image_path_ = true ;
        obj.singularity_detection_image_path_ = DLBackEndClass.legacy_default_singularity_image_path_for_detect ;
      end  
    end
  end

  methods   % private by convention
    function stopEc2InstanceIfNeeded_(obj)
      aws = obj.awsec2;
      % DEBUGAWS: Stopping the AWS instance takes too long when debugging.
      if aws.isInDebugMode ,
        return
      end
      fprintf('Stopping AWS EC2 instance %s...\n',aws.instanceID);
      tfsucc = aws.stopInstance();
      if ~tfsucc
        warningNoTrace('Failed to stop AWS EC2 instance %s.',aws.instanceID);
      end
    end  % function    
  end  % methods block

  methods
    function result = get.isInAwsDebugMode(obj)
      result = obj.awsec2.isInDebugMode ;
    end

    function set.isInAwsDebugMode(obj, value)
      obj.awsec2.isInDebugMode = value ;
    end    
  end  % methods block

  methods
    function clearRegisteredJobs(obj)
      % Clear all registered jobs
      obj.syscmds_ = cell(0,1) ;
      obj.cmdfiles_ = cell(0,1) ;
      obj.logcmds_ = cell(0,1) ;   
    end

    function registerTrainingJob(backend, dmcjob, deeptracker, gpuids, aptroot, do_just_generate_db)
      % Register a single training job with the backend, for later spawning via
      % spawnRegisteredJobs().
      ignore_local = (backend.type == DLBackEnd.Bsub) ;  % whether to pass the --ignore_local options to APTInterface.py
      basecmd = APTInterf.trainCodeGenBase(dmcjob,'ignore_local',ignore_local,'aptroot',aptroot,'do_just_generate_db',do_just_generate_db);
      args = determineArgumentsForSpawningJob(backend,deeptracker,gpuids,dmcjob,aptroot,'train');
      syscmd = wrapCommandToBeSpawnedForBackend(backend,basecmd,args{:});
      cmdfile = DeepModelChainOnDisk.getCheckSingle(dmcjob.trainCmdfileLnx());

      if backend.type == DLBackEnd.Docker,
        containerName = DeepModelChainOnDisk.getCheckSingle(dmcjob.trainContainerName);
        logfile = DeepModelChainOnDisk.getCheckSingle(dmcjob.trainLogLnx);
        logcmd = generateLogCommandForDockerBackend(backend,containerName,logfile);
      else
        logcmd = '' ;
      end

      % Add all the commands to the registry
      backend.syscmds_{end+1,1} = syscmd ;
      backend.logcmds_{end+1,1} = logcmd ;
      backend.cmdfiles_{end+1,1} = cmdfile ;
    end

    function registerTrackingJob(backend, totrackinfojob, deeptracker, gpuids, aptroot, track_type)
      % Register a single tracking job with the backend, for later spawning via
      % spawnRegisteredJobs().
      ignore_local = (backend.type == DLBackEnd.Bsub) ;  % whether to pass the --ignore_local options to APTInterface.py
      basecmd = APTInterf.trackCodeGenBase(totrackinfojob,'ignore_local',ignore_local,'aptroot',aptroot,'track_type',track_type);
      args = determineArgumentsForSpawningJob(backend, deeptracker, gpuids, totrackinfojob, aptroot, 'track') ;
      syscmd = wrapCommandToBeSpawnedForBackend(backend, basecmd, args{:}) ;
      cmdfile = DeepModelChainOnDisk.getCheckSingle(totrackinfojob.cmdfile) ;

      if backend.type == DLBackEnd.Docker,
        containerName = totrackinfojob.containerName;
        logfile = totrackinfojob.logfile;
        logcmd = generateLogCommandForDockerBackend(backend,containerName,logfile);
      else
        logcmd = '' ;
      end
    
      % Add all the commands to the registry
      backend.syscmds_{end+1,1} = syscmd ;
      backend.logcmds_{end+1,1} = logcmd ;
      backend.cmdfiles_{end+1,1} = cmdfile ;
    end

    function [tfSucc, jobID] = spawnRegisteredJobs(obj, varargin)
      % Spawn all the jobs that have been previously registered.  Jobs should be
      % either all tracking jobs or all training jobs.  
      [jobdesc, do_call_apt_interface_dot_py] = myparse( ...
        varargin, ...
        'jobdesc', 'job', ...
        'do_call_apt_interface_dot_py', true) ;
      syscmds = obj.syscmds_ ;
      logcmds = obj.logcmds_ ;
      cmdfiles = obj.cmdfiles_ ;
      if ~isempty(cmdfiles),
        obj.writeCmdToFile(syscmds,cmdfiles,jobdesc);
      end
      njobs = numel(syscmds);
      tfSucc = false(1,njobs);
      tfSuccLog = true(1,njobs);
      jobID = cell(1,njobs);
      for ijob=1:njobs,
        syscmd = syscmds{ijob} ;
        fprintf(1,'%s\n',syscmd);
        if do_call_apt_interface_dot_py ,
%           if obj.type == DLBackEnd.Conda,
%             [jobID{ijob},st,res] = parfevalsystem(syscmd);
%             tfSucc(ijob) = st == 0;
%           else
          [st,res] = apt.syscmd(syscmd, 'failbehavior', 'silent');
          tfSucc(ijob) = (st == 0) ;
          if tfSucc(ijob),
            jobID{ijob} = obj.parseJobID(res);
          end
        else
          % Pretend it's a failure, for expediency.
          tfSucc(ijob) = false ;
          res = 'do_call_apt_interface_dot_py is false' ;
        end
        if ~tfSucc(ijob),
          warning('Failed to spawn %s %d:\n%s',jobdesc,ijob,res);
        else
          jobid = jobID{ijob};
          if isnumeric(jobid),
            jobidstr = num2str(jobid);
          elseif isa(jobid, 'parallel.FevalFuture') ,
            jobidstr = num2str(jobid.ID) ;
          else
            % hopefully the jobid is a string
            jobidstr = jobid ;
          end
          fprintf('%s %d spawned, ID = %s\n\n',jobdesc,ijob,jobidstr);
        end
        if numel(logcmds) >= ijob ,
          logcmd = logcmds{ijob} ;
          if ~isempty(logcmd) ,
            fprintf(1,'%s\n',logcmd);
            [st2,res2] = apt.syscmd(logcmd, 'failbehavior', 'silent');
            tfSuccLog(ijob) = (st2==0) ;
            if ~tfSuccLog(ijob),
              warning('Failed to spawn logging for %s %d: %s.',jobdesc,ijob,res2);
            end
          end
        end
      end
    end  % function
  end  % methods
end  % classdef
