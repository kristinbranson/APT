classdef BgTrainWorkerObjBsub < BgTrainWorkerObj
  
  methods
    
    function obj = BgTrainWorkerObjBsub(dlLblFile,jobID,logFiles)
      obj@BgTrainWorkerObj(dlLblFile,jobID);
      
      assert(iscellstr(logFiles) && numel(logFiles)==obj.nviews);
      obj.artfctLogs = logFiles;

      [obj.artfctTrainDataJson,obj.artfctFinalIndex,obj.artfctErrFile] = ...
        arrayfun(@obj.trainMonitorArtifacts,1:obj.nviews,'uni',0);
    end
    
    function [json,finalindex,errfile] = trainMonitorArtifacts(obj,ivw)
      fprintf(2,'Should prob match AWS\n');
      
      cacheDir = obj.sPrm.CacheDir;
      projvw = sprintf('%s_view%d',obj.projname,ivw-1); % !! cacheDirs are 0-BASED
      subdir = fullfile(cacheDir,projvw,obj.jobID);
      
      json = sprintf('%s_pose_unet_traindata.json',projvw);
      json = fullfile(subdir,json);
      
      finaliter = obj.sPrm.dl_steps;
      finalindex = sprintf('%s_pose_unet-%d.index',projvw,finaliter);
      finalindex = fullfile(subdir,finalindex);
      
      errfile = DeepTracker.dlerrGetErrFile(obj.jobID);
    end
    
    function tf = fileExists(~,file)
      tf = exists(file,'file')>0;
    end
    
    function tf = errFileExistsNonZeroSize(~,errFile)
      tf = BgTrainWorkerObjBsub.errFileExistsNonZeroSizeStc(errFile);
    end
        
    function s = fileContents(~,file)
      lines = readtxtfile(file);
      s = sprintf('%s\n',lines{:});
    end
    
  end
    
  methods (Static)
    function tfErrFileErr = errFileExistsNonZeroSizeStc(errFile)
      tfErrFileErr = exist(errFile,'file')>0;
      if tfErrFileErr
        direrrfile = dir(errFile);
        tfErrFileErr = direrrfile.bytes>0;
      end
    end
  end
  
end