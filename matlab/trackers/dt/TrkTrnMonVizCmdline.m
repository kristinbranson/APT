classdef TrkTrnMonVizCmdline < handle
  
  methods
    
    function obj = TrkTrnMonVizCmdline(varargin)
    end
    
    function [tfSucc,msg] = resultsReceived(obj, res)
      % Callback executed when new result received from training monitor BG
      % worker
      %
      % trnComplete: scalar logical, true when all views done
      
      fprintf(1,' ### Results size: %s\n',mat2str(size(res)));
      for ivw=1:numel(res) 
        fprintf(1,'### Elem %d\n',ivw);
        disp(res(ivw));
        
%         fprintf(1,'View%d: jsonPresent: %d. ',ivw,res(ivw).jsonPresent);
%         if res(ivw).tfUpdate
%           fprintf(1,'New training iter: %d.\n',res(ivw).lastTrnIter);
%         elseif res(ivw).jsonPresent
%           fprintf(1,'No update, still on iter %d.\n',res(ivw).lastTrnIter);
%         else
%           fprintf(1,'\n');
%         end
      end  
      
      tfSucc = true;
      msg = '';
    end
    
  end
  
end
