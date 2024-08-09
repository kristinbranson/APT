classdef TrkTrnMonVizSimpleStore < handle
  
  properties
    dat
  end
  
  methods
    
    function obj = TrkTrnMonVizSimpleStore(varargin)
      obj.dat = cell(0,1);
    end
    
    function [tfSucc,msg] = resultsReceived(obj,sRes)
      % Callback executed when new result received from training monitor BG
      % worker

      
      res = sRes.result;
      obj.dat{end+1,1} = res;
      fprintf(1,'Res received...\n');
            
      tfSucc = true;
      msg = '';
    end
    
  end
  
end
