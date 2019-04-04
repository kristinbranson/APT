classdef DLBackEndClass < matlab.mixin.Copyable
  % Design unclear but good chance this is a thing
  %
  % This thing (maybe) specifies a physical machine/server along with a 
  % DLBackEnd:
  % * DLNetType: what DL net is run
  % * DLBackEndClass: where/how DL was run
  %
  % TODO: this should be named 'DLBackEnd' and 'DLBackEnd' should be called
  % 'DLBackEndType' or something.
  %
  % copy() produces a deep copy (new/separate awsec2)
  
  properties
    type  % scalar DLBackEnd
    
    % scalar logical. if true, backend runs code in APT.Root/deepnet. This
    % path must be visible in the backend or else.
    deepnetrunlocal = true; 
  end
  properties (NonCopyable)
    awsec2 % used only for type==AWS
  end    
 
  methods
    
    function obj = DLBackEndClass(ty)
      obj.type = ty;
    end
    
    function [tf,reason] = getReadyTrainTrack(obj)
      if obj.type==DLBackEnd.AWS
        aws = obj.awsec2;
        
        tf = ~isempty(aws);
        if ~tf
          reason = 'AWS EC2 instance is not configured.';
          return;
        end        
        
        [tfexist,tfrunning] = aws.inspectInstance;
        tf = tfrunning;
        if ~tf
          reason = sprintf('AWS EC2 instance %s is not running.',aws.instanceID);
          return;
        end
        
        reason = '';
      else
        tf = true;
        reason = '';
      end
    end
    
    function s = prettyName(obj)      
      switch obj.type,
        case DLBackEnd.Bsub,
          s = 'JRC Cluster';
        case DLBackEnd.Docker,
          s = 'Local';
        otherwise
          s = char(obj.type);
      end
    end
    
%     function tf = filesysAreCompatible(obj,obj2)
%       assert(isscalar(obj) && isscalar(obj2));
%     end
  end
  
  methods (Access = protected)
    function cpObj = copyElement(obj)
      % Overloaded to deep copy .awsec2      
      cpObj = copyElement@matlab.mixin.Copyable(obj);
      if ~isempty(obj.awsec2)
        assert(isa(obj.awsec2,'AWSec2') && isscalar(obj.awsec2));
        cpObj.awsec2 = obj.awsec2.copy();
      end
    end
  end
  
end
    