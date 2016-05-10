classdef IMP < impoint

  properties
    deleteFcn % called during destruction
  end
  
  methods    
    function obj = IMP(varargin)
      obj@impoint(varargin{:});
    end    
    function delete(obj)
      if ~isempty(obj.deleteFcn)
        fcn = obj.deleteFcn;
        obj.deleteFcn = [];
        fcn();
      end
    end
  end
end