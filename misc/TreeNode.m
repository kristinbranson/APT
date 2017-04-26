classdef TreeNode < handle

  properties
    Data
    Children % vector of TreeNodes
  end
  
  methods
    
    function obj = TreeNode(dat)
      obj.Children = TreeNode.empty(1,0);
      obj.Data = dat;
    end
    
    function traverse(t,fcn)
      assert(isscalar(t));
      fcn(t);
      c = t.Children;
      arrayfun(fcn,c);
    end
    
  end
  
end  