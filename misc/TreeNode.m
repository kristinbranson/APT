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
    
    function s = structize(t)
      
      s = nst(t,struct());
      function s = nst(t,s)
        fld = t.Data.Field;
        val = t.Data.Value;
        s.(fld) = val;
        cs = t.Children;
        for i=1:numel(cs)
          s.(fld) = nst(cs(i),s.(fld));
        end
      end
    end
    
  end
  
end  