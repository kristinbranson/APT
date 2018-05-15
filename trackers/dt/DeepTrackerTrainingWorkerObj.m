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
      
      obj.trnLogLastStep = repmat(-1,1,obj.nviews);
    end
    
    function sRes = compute(obj)
      % sRes: [nviewx1] struct array. 
      
      cacheDir = obj.sPrm.CacheDir;
      %[~,lblfileS] = fileparts(obj.dlLblFile);

      % Atm we read the json for every view and see if it has been updated.
      sRes = struct(...
      	   'jsonPath',cell(obj.nviews,1),... % char, full path to json trnlog being polled
      	   'jsonPresent',[],... % true if file exists. if false, remaining fields are indeterminate
	   'lastTrnIter',[],... % last known training iter for this view. Could be eg -1 or 0 if no iters avail yet.
	   'tfUpdate',[],... % true if the current read represents an updated training iter.  
	   'contents',[]); % if tfupdate is true, this can contain all json contents.
      for ivw=1:obj.nviews
        projvw = sprintf('%s_view%d',obj.projname,ivw-1);
        subdir = fullfile(cacheDir,projvw,obj.jobID);
        json = sprintf('%s_pose_unet_traindata.json',projvw);
        json = fullfile(subdir,json);
	sRes(ivw).jsonPath = json;
	sRes(ivw).jsonPresent = exist(json,'file')>0;
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
  end
  
end