classdef BgTrainMonitorAWS < BgTrainMonitor
  
  properties
    awsEc2 % scalar handle AWSec2
    remotePID % view1 remote PID
  end
  
  methods
    
    function obj = BgTrainMonitorAWS
      obj@BgTrainMonitor();
    end
    
    function prepareHook(obj,trnMonVizObj,bgWorkerObj)
      obj.awsEc2 = bgWorkerObj.awsEc2;
    end    
    
    function bgTrnResultsReceivedHook(obj,sRes)
      % TODO: commong code factor me with BgTrainMonitorBsub
      
      errOccurred = any([sRes.result.errFileExists]);
      if errOccurred
        obj.stop();
        
        fprintf(1,'Error occurred during training:\n');
        errFile = sRes.result(1).errFile; % currently, errFiles same for all views
        fprintf(1,'\n### %s\n\n',errFile);
        errContents = obj.bgWorkerObj.remoteFileContents(errFile,'dispcmd',true);
        disp(errContents);
        fprintf(1,'\n\n. You may need to manually kill any running DeepLearning process.\n');
        
        % monitor plot stays up; reset not called etc
      end
      
      for i=1:numel(sRes.result)
        if sRes.result(i).logFileErrLikely
          obj.stop();
          
          fprintf(1,'Error occurred during training:\n');
          errFile = sRes.result(i).logFile;
          fprintf(1,'\n### %s\n\n',errFile);
          errContents = obj.bgWorkerObj.remoteFileContents(errFile,'dispcmd',true);
          disp(errContents);
          fprintf(1,'\n\n. You may need to manually kill any running DeepLearning process.\n');
          
          % monitor plot stays up; bgTrnReset not called etc
        end
      end
      
      trnComplete = all([sRes.result.trainComplete]);
      if trnComplete
        obj.stop();
        % % monitor plot stays up; reset not called etc
        fprintf('Training complete at %s.\n',datestr(now));
        fprintf('COPY/DO STUFF!!\n');
      end
    end
   
    function killRemoteProcess(obj)
      if isempty(obj.remotePID)
        error('Unknown PID for remote process.');
      end
      
      cmdremote = sprintf('kill %d',obj.remotePID);
      [tfsucc,res] = obj.awsEc2.cmdInstance(cmdremote,'dispcmd',true);
      if tfsucc
        fprintf('Kill command sent.\n\n');
      else
        error('Kill command failed.');
      end

    end
    
  end
  
end