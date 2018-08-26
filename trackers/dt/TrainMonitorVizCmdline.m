classdef TrainMonitorVizCmdline < handle
  
  methods
    
    function obj = TrainMonitorVizCmdline(varargin)
    end
    
    function resultsReceived(obj,sRes)
      % Callback executed when new result received from training monitor BG
      % worker
      %
      % trnComplete: scalar logical, true when all views done
      
      res = sRes.result;
%       tfAnyUpdate = false;
      
%       disp('foo');
      for ivw=1:numel(res)
        fprintf(1,'View%d: jsonPresent: %d. ',ivw,res(ivw).jsonPresent);
        if res(ivw).tfUpdate
          fprintf(1,'New training iter: %d.\n',res(ivw).lastTrnIter);
        elseif res(ivw).jsonPresent
          fprintf(1,'No update, still on iter %d.\n',res(ivw).lastTrnIter);
        else
          fprintf(1,'\n');
        end
        disp(res(ivw));
      end      
%       if isempty(obj.resLast) || tfAnyUpdate
%         obj.resLast = res;
%       end
    end
    
  end
  
end

