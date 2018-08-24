classdef BgTrainMonitorBsub < BgTrainMonitor
  
  methods
    
    function obj = BgTrainMonitorBsub
      obj@BgTrainMonitor();
    end
    
    function bgTrnResultsReceivedHook(obj,sRes)
      errOccurred = any([sRes.result.errFileExists]);
      if errOccurred
        obj.stop();
        
        fprintf(1,'Error occurred during training:\n');
        errFile = sRes.result(1).errFile; % currently, errFiles same for all views
        fprintf(1,'\n### %s\n\n',errFile);
        type(errFile);
        fprintf(1,'\n\n. You may need to manually kill any running DeepLearning process.\n');
        
        % monitor plot stays up; bgTrnReset not called etc
      end
      
      for i=1:numel(sRes.result)
        if sRes.result(i).bsubLogFileErrLikely
          obj.stop();
          
          fprintf(1,'Error occurred during training:\n');
          errFile = sRes.result(i).bsubLogFile;
          fprintf(1,'\n### %s\n\n',errFile);
          type(errFile);
          fprintf(1,'\n\n. You may need to manually kill any running DeepLearning process.\n');
          
          % monitor plot stays up; bgTrnReset not called etc
        end
      end
      
      trnComplete = all([sRes.result.trainComplete]);
      if trnComplete
        obj.stop();
        % % monitor plot stays up; bgTrnReset not called etc
        fprintf('Training complete at %s.\n',datestr(now));
      end
    end
  end
  
end