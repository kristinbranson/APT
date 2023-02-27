classdef DeepModelChainReader < matlab.mixin.Copyable
  methods (Abstract)
    tf = getModelIsRemote(obj)
    [maxiter,idx] = getMostRecentModel(obj,dmc,varargin)
    lsProjDir(obj,dmc,idx)
    lsModelChainDir(obj,dmc,varargin)
  end
  methods
    function prepareBg(obj) %#ok<MANU>
      % 'Detach' a reader for use in bg processes
      % Typically you would deepcopy the reader before calling this
      
      % pass      
    end
  end
  methods (Static)
    function obj = createFromBackEnd(be)
      % be: DLBackEndClass
      switch be.type
        case DLBackEnd.AWS
          obj = DeepModelChainReaderAWS(be.awsec2);
        otherwise
          obj = DeepModelChainReaderLocal();
      end
    end
  end
end