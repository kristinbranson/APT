classdef DeepModelChainReader < handle
  methods (Abstract)
    tf = getModelIsRemote(obj)
    maxiter = getMostRecentModel(obj,dmc)
    lsProjDir(obj,dmc)
    lsModelChainDir(obj,dmc)
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