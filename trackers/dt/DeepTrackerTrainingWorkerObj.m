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
    jobID % char
    
    trnLogLastStep; % [nview] int. most recent last step from training json logs
  end
  
  methods
    function obj = DeepTrackerTrainingWorkerObj(dlLblFile,jobID)
      obj.posetfroot = APT.getpathdl;

      lbl = load(dlLblFile,'-mat');      
      obj.nviews = lbl.cfg.NumViews;
      obj.sPrm = lbl.trackerDeepData.sPrm; % .sPrm guaranteed to match dlLblFile
      obj.dlLblFile = dlLblFile;
      obj.jobID = jobID;
      
      obj.trnLogLastStep = repmat(-1,1,obj.nviews);
    end
    
    function sRes = compute(obj)
      % sRes: [nviewx1] array. Each el is either [], indicating no change
      % in the training json/log, or it is a struct containing the contents
      % of that json.
      
      cacheDir = obj.sPrm.CacheDir;
      [~,lblfileS] = fileparts(obj.dlLblFile);

      % Atm we read the json for every view and see if it has been updated.
      sRes = cell(obj.nviews,1);
      for ivw=1:obj.nviews
        jobidvw = sprintf('%s_view%d',obj.jobID,ivw);
        subdir = fullfile(cacheDir,jobidvw,lblfileS);
        json = sprintf('%s_pose_unet_traindata.json',jobidvw);
        json = fullfile(subdir,json);
        if exist(json,'file')>0
          json = fileread(json);
          trnLog = jsondecode(json);
          if trnLog.step(end)>obj.trnLogLastStep(ivw)
            sRes{ivw} = trnLog;
            obj.trnLogLastStep(ivw) = trnLog.step(end);
          end
        end
      end      
    end
  end
  
end