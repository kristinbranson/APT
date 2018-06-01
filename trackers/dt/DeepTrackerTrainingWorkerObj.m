classdef DeepTrackerTrainingWorkerObj < handle
  % Object deep copied onto BG Training worker. To be used with
  % BGWorkerContinuous
  %
  % Responsibilities:
  % - Poll filesystem for training updates
  
  properties
    posetfroot % root dir of poseTF checkout
    
    nviews
    sPrm % parameter struct
    dlLblFile % fullpath to stripped lblfile
    projname
    jobID % char
    
    artfctTrainDataJson % [nview] cellstr of fullpaths to training data jsons
    artfctFinalIndex % [nview] cellstr of fullpaths to final training .index file
    
    trnLogLastStep; % [nview] int. most recent last step from training json logs
  end
  
  methods
    function obj = DeepTrackerTrainingWorkerObj(dlLblFile,jobID)
      obj.posetfroot = APT.getpathdl;
      
      lbl = load(dlLblFile,'-mat');
      obj.nviews = lbl.cfg.NumViews;
      obj.sPrm = lbl.trackerDeepData.sPrm; % .sPrm guaranteed to match dlLblFile
      obj.dlLblFile = dlLblFile;
      obj.projname = lbl.projname;
      obj.jobID = jobID;
      
      [obj.artfctTrainDataJson,obj.artfctFinalIndex] = ...
        arrayfun(@obj.trainMonitorArtifacts,1:obj.nviews,'uni',0);
      
      obj.trnLogLastStep = repmat(-1,1,obj.nviews);
    end
    
    function sRes = compute(obj)
      % sRes: [nviewx1] struct array.
            
      % - Read the json for every view and see if it has been updated.
      % - Check for completion 
      sRes = struct(...
        'jsonPath',cell(obj.nviews,1),... % char, full path to json trnlog being polled
        'jsonPresent',[],... % true if file exists. if false, remaining fields are indeterminate
        'lastTrnIter',[],... % (only if jsonPresent==true) last known training iter for this view. Could be eg -1 or 0 if no iters avail yet.
        'tfUpdate',[],... % (only if jsonPresent==true) true if the current read represents an updated training iter.
        'contents',[],... % (only if jsonPresent==true) if tfupdate is true, this can contain all json contents.
        'trainCompletePath',[],... % char, full path to artifact indicating train complete
        'trainComplete',[]... % true if trainCompletePath exists
        ); % 
      for ivw=1:obj.nviews
        json = obj.artfctTrainDataJson{ivw};
        finalindex = obj.artfctFinalIndex{ivw};
        
        sRes(ivw).jsonPath = json;
        sRes(ivw).jsonPresent = exist(json,'file')>0;
        sRes(ivw).trainCompletePath = finalindex;
        sRes(ivw).trainComplete = exist(finalindex,'file')>0;
        
        if sRes(ivw).jsonPresent
          json = fileread(json);
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
    
    function [json,finalindex] = trainMonitorArtifacts(obj,ivw)
      cacheDir = obj.sPrm.CacheDir;
      projvw = sprintf('%s_view%d',obj.projname,ivw-1);
      subdir = fullfile(cacheDir,projvw,obj.jobID);
      
      json = sprintf('%s_pose_unet_traindata.json',projvw);
      json = fullfile(subdir,json);
      
      finaliter = obj.sPrm.unet_steps;
      finalindex = sprintf('%s_pose_unet-%d.index',projvw,finaliter);
      finalindex = fullfile(subdir,finalindex);
    end
        
  end
  
end