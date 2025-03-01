classdef TrkTrnMonVizSimpleStore < handle
  
  properties
    dat
  end
  
  methods
    
    function obj = TrkTrnMonVizSimpleStore(varargin)
      obj.dat = cell(0,1);
    end
    
    function [tfSucc,msg] = resultsReceived(obj,res)
      % Callback executed when new result received from training monitor BG
      % worker
      
      obj.dat{end+1,1} = res;
      fprintf(1,'Res received...\n');
            
      tfSucc = true;
      msg = '';
    end
    
  end
  
end
