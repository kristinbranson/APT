classdef DLBackEndClass < handle
  % Placeholder class, future design unclear
  
  properties
    type  % scalar DLBackEnd
    awsec2 % used only for type==AWS
  end
 
  methods
    
    function obj = DLBackEndClass(ty)
      obj.type = ty;
    end
    
    function [tf,reason] = getReadyTrainTrack(obj)
      if obj.type==DLBackEnd.AWS
        tf = ~isempty(obj.awsec2);
        if ~tf
          reason = 'AWS EC2 instance is not configured.';
        else
          reason = '';
        end
      else
        tf = true;
        reason = '';
      end
    end      
 
  end
  
end
    