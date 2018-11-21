classdef BgTrainWorkerObjBsub < BgTrainWorkerObj
  
  properties
    jobID % [nview] bsub jobID
  end
  
  methods
    
    function obj = BgTrainWorkerObjBsub(nviews,dmcs)
      obj@BgTrainWorkerObj(nviews,dmcs);      
    end
    
    function tf = fileExists(~,file)
      tf = exist(file,'file')>0;
    end
    
    function tf = errFileExistsNonZeroSize(~,errFile)
      tf = BgTrainWorkerObjBsub.errFileExistsNonZeroSizeStc(errFile);
    end
        
    function s = fileContents(~,file)
      if exist(file,'file')==0
        s = '<file does not exist>';
      else
        lines = readtxtfile(file);
        s = sprintf('%s\n',lines{:});
      end
    end
    
    function killProcess(obj)
%       if ~obj.isRunning
%         error('Training is not in progress.');
%       end
%       if isempty(obj.jobID) || isnan(obj.jobID)
%          error('jobID is unset.');
%       end
      
      dmcs = obj.dmcs;
      killfiles = {dmcs.killTokenLnx};
      jobids = obj.jobID;
      nvw = obj.nviews;
      assert(isequal(nvw,numel(jobids),numel(killfiles)));
      
      for ivw=1:nvw
        bkillcmd = sprintf('bkill %d',jobids(ivw));
        bkillcmd = DeepTracker.codeGenSSHGeneral(bkillcmd,'bg',false);
        fprintf(1,'%s\n',bkillcmd);
        [st,res] = system(bkillcmd);
        if st~=0
          warningNoTrace('Bkill command failed: %s',res);          
        end
      end
      
      for ivw=1:nvw
        fcn = makeBsubJobKilledPollFcn(jobids(ivw));
        iterWaitTime = 1;
        maxWaitTime = 12;
        tfsucc = waitforPoll(fcn,iterWaitTime,maxWaitTime);

        if ~tfsucc
          warningNoTrace('Could not confirm that bsub job was killed.');
        else
          % touch KILLED tokens i) to record kill and ii) for bgTrkMonitor to 
          % pick up
          
          kfile = killfiles{ivw};
          touchcmd = sprintf('touch %s',kfile);
          touchcmd = DeepTracker.codeGenSSHGeneral(touchcmd,'bg',false);
          [st,res] = system(touchcmd);
          if st~=0
            warningNoTrace('Failed to create KILLED token: %s',kfile);
          else
            fprintf('Created KILLED token: %s.\nPlease wait for your training monitor to acknowledge the kill!\n',kfile);
          end          
        end

        % bgTrnMonitor should pick up KILL tokens and stop bg trn monitoring
      end
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


function fcn = makeBsubJobKilledPollFcn(jobID)

pollcmd = sprintf('bjobs -o stat -noheader %d',jobID);
pollcmd = DeepTracker.codeGenSSHGeneral(pollcmd,'bg',false);
 
fcn = @lcl;

  function tf = lcl
    % returns true when jobID is killed
    %disp(pollcmd);
    [st,res] = system(pollcmd);
    if st==0
      tf = isempty(regexp(res,'RUN','once'));      
    else
      tf = false;
    end
  end
end