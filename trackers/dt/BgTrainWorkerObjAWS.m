classdef BgTrainWorkerObjAWS < BgTrainWorkerObj
  
  properties
    awsEc2 % Instance of AWSec2
  end
  
  methods

    function obj = BgTrainWorkerObjAWS(dlLblFile,jobID,cacheRemoteRel,...
        logfilesremote,awsec2)
  
      obj@BgTrainWorkerObj(dlLblFile,jobID);
      
      obj.artfctLogs = logfilesremote;
      [obj.artfctTrainDataJson,obj.artfctFinalIndex,obj.artfctErrFile] = ...
        arrayfun(@(ivw)obj.trainMonitorArtifacts(cacheRemoteRel,ivw),...
        1:obj.nviews,'uni',0);
            
      obj.awsEc2 = awsec2;
    end
        
    function [json,finalindex,errfile] = trainMonitorArtifacts(obj,...
        cacheRemoteRel,ivw)
%       cacheDir = obj.sPrm.CacheDir;
%       [~,cacheDirS] = fileparts(cacheDir);
      
      projvw = sprintf('%s_view%d',obj.projname,ivw-1); % !! cacheDirs are 0-BASED
%       subdir = fullfile('/home/ubuntu',cacheRemoteRel,projvw,obj.jobID);  
      subdir = fullfile('/home/ubuntu',cacheRemoteRel);
      
%       json = sprintf('%s_pose_unet_traindata.json',projvw);
      json = 'traindata.json'; % AL AWS testing 20180826: what happens with multiview?
      json = fullfile(subdir,json);
      json = FSPath.standardPathChar(json);
      
      finaliter = obj.sPrm.dl_steps;
      finalindex = sprintf('%s_pose_unet-%d.index',projvw,finaliter);
      finalindex = fullfile(subdir,finalindex);
      finalindex = FSPath.standardPathChar(finalindex);
      
      errfile = DeepTracker.dlerrGetErrFile(obj.jobID,'/home/ubuntu');
      errfile = FSPath.standardPathChar(errfile);
    end
    
    function tf = fileExists(obj,f)
      tf = obj.awsEc2.remoteFileExists(f);
    end
    
    function tf = errFileExistsNonZeroSize(obj,errFile)
      tf = obj.awsEc2.remoteFileExists(errFile,'reqnonempty',true);
    end    
    
    function s = fileContents(obj,f)
      s = obj.awsEc2.remoteFileContents(f);
    end
        
  end
    
end