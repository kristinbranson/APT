classdef BgTrainWorkerObjAws < handle
  % Object deep copied onto BG Training worker. To be used with
  % BGWorkerContinuous.
  %
  % BGTrainWorkerObjAws runs in the bg of the client machine, but 
  % i) accesses local filesys for .lbl files, pem keys, etc;
  % ii) has an AWSec2 object for communicating with remote EC2 instance
  %
  % Responsibilities:
  % - Poll remote (AWS EC2) filesystem for training updates
  
  properties
    nviews
    sPrm % parameter struct
    projname
    jobID % char
    
    awsEc2 % Instace of AWSec2
    
    artfctTrainDataJson % [nview] cellstr of fullpaths to training data jsons
    artfctFinalIndex % [nview] cellstr of fullpaths to final training .index file
    artfctErrFile % [nview] cellstr of fullpaths to DL errfile
    
    trnLogLastStep; % [nview] int. most recent last step from training json logs
  end
  
  methods

    function obj = BgTrainWorkerObjAws(dlLblFile,jobID,awsec2)
      lbl = load(dlLblFile,'-mat');
      obj.nviews = lbl.cfg.NumViews;
      obj.sPrm = lbl.trackerDeepData.sPrm; % .sPrm guaranteed to match dlLblFile
      obj.projname = lbl.projname;
      obj.jobID = jobID;
      
      obj.awsEc2 = awsec2;
      
      [obj.artfctTrainDataJson,obj.artfctFinalIndex,obj.artfctErrFile] = ...
        arrayfun(@obj.trainMonitorArtifacts,1:obj.nviews,'uni',0);
      
      obj.trnLogLastStep = repmat(-1,1,obj.nviews);
    end
    
    function sRes = compute(obj,varargin)
      % sRes: [nviewx1] struct array.
      
      verbose = myparse(varargin,...
        'verbose',false);
            
      % - Read the json for every view and see if it has been updated.
      % - Check for completion 
      sRes = struct(...
        'jsonPath',cell(obj.nviews,1),... % char, full path to json trnlog being polled
        'jsonPresent',[],... % true if file exists. if false, remaining fields are indeterminate
        'lastTrnIter',[],... % (only if jsonPresent==true) last known training iter for this view. Could be eg -1 or 0 if no iters avail yet.
        'tfUpdate',[],... % (only if jsonPresent==true) true if the current read represents an updated training iter.
        'contents',[],... % (only if jsonPresent==true) if tfupdate is true, this can contain all json contents.
        'trainCompletePath',[],... % char, full path to artifact indicating train complete
        'trainComplete',[],... % true if trainCompletePath exists
        'errFile',[],... % char, full path to DL err file
        'errFileExists',[]... % true of errFile exists and has size>0
        );
      for ivw=1:obj.nviews
        json = obj.artfctTrainDataJson{ivw};
        finalindex = obj.artfctFinalIndex{ivw};
        errFile = obj.artfctErrFile{ivw};
        
        sRes(ivw).jsonPath = json;
        sRes(ivw).jsonPresent = obj.remoteFileExists(json,'dispcmd',verbose);
        sRes(ivw).trainCompletePath = finalindex;
        sRes(ivw).trainComplete = obj.remoteFileExists(finalindex,'dispcmd',verbose);
        sRes(ivw).errFile = errFile;
        sRes(ivw).errFileExists = obj.remoteFileExists(errFile,'reqnonempty',true,'dispcmd',verbose);
        
        if sRes(ivw).jsonPresent
          json = obj.remoteFileContents(json,'dispcmd',verbose);
          trnLog = jsondecode(json);
          lastKnownStep = obj.trnLogLastStep(ivw);
          newStep = trnLog.step(end);
          tfupdate = newStep>lastKnownStep;
          sRes(ivw).tfUpdate = tfupdate;
          if tfupdate
            sRes(ivw).lastTrnIter = newStep;
            obj.trnLogLastStep(ivw) = newStep;
            sRes(ivw).contents = trnLog;
          else
            sRes(ivw).lastTrnIter = lastKnownStep;
          end
        end
      end
    end
    
    function [json,finalindex,errfile] = trainMonitorArtifacts(obj,ivw)
      cacheDir = obj.sPrm.CacheDir;
      [~,cacheDirS] = fileparts(cacheDir);
      
      projvw = sprintf('%s_view%d',obj.projname,ivw-1); % !! cacheDirs are 0-BASED
      subdir = fullfile('/home/ubuntu',cacheDirS,projvw,obj.jobID);
      
      json = sprintf('%s_pose_unet_traindata.json',projvw);
      json = fullfile(subdir,json);
      json = FSPath.standardPathChar(json);
      
      finaliter = obj.sPrm.dl_steps;
      finalindex = sprintf('%s_pose_unet-%d.index',projvw,finaliter);
      finalindex = fullfile(subdir,finalindex);
      finalindex = FSPath.standardPathChar(finalindex);
      
      errfile = DeepTracker.dlerrGetErrFile(obj.jobID,'/home/ubuntu');
      errfile = FSPath.standardPathChar(errfile);
    end

    function [tfEFE,errFile] = errfileExists(obj)
      assert(false,'TODO');
    end

    function tf = remoteFileExists(obj,f,varargin)
      [reqnonempty,dispcmd] = myparse(varargin,...
        'reqnonempty',false,...
        'dispcmd',false...
        );

      if reqnonempty
        script = '~/APT/misc/fileexistsnonempty.sh';
      else
        script = '~/APT/misc/fileexists.sh';
      end
      cmdremote = sprintf('%s %s',script,f);
      [~,res] = obj.awsEc2.cmdInstance(cmdremote,...
        'dispcmd',dispcmd,'harderronfail',true); 
      tf = res(1)=='y';      
    end

    function s = remoteFileContents(obj,f,varargin)
      dispcmd = myparse(varargin,...
        'dispcmd',false);
      
      cmdremote = sprintf('cat %s',f);
      [~,res] = obj.awsEc2.cmdInstance(cmdremote,...
        'dispcmd',dispcmd,'harderronfail',true); 
        
      s = res;
    end
    
  end
  
  methods (Static)
    function tfErrFileErr = errFileExistsNonZeroSize(errFile)
      errFileExist = exist(errFile,'file')>0;
      if errFileExist
        direrrfile = dir(errFile);
        tfErrFileErr = direrrfile.bytes>0;
      else
        tfErrFileErr = false;
      end
    end
  end
  
end