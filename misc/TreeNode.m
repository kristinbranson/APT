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
      % Convert a Tree (Data.Value fields only) to a struct
    
      assert(isscalar(t));
      s = nst(t,struct());
      function s = nst(t,s)
        fld = t.Data.Field;
        val = t.Data.Value;
        s.(fld) = val;
        cs = t.Children;
        for j=1:numel(cs)
          s.(fld) = nst(cs(j),s.(fld));
        end
      end
    end
    
    function structapply(t,s)
      % Apply values from a structure to Data.Value fields of leaf nodes
      % 
      % t: vector of TreeNodes
      % s: struct
      
      fnS = fieldnames(s);
      fnT = arrayfun(@(x)x.Data.Field,t,'uni',0);
      for f=fnS(:)',f=f{1}; %#ok<FXSET>
        tf = strcmpi(f,fnT);
        if any(tf)
          assert(nnz(tf)==1);
          node = t(tf);
          val = s.(f);
          if isstruct(val)
            % val is a struct; node must be a non-leafnode
            structapply(node.Children,val);
          else            
            assert(isempty(node.Children));
            node.Data.Value = val;
          end
        else
          warningNoTrace('TreeNode:field',...
            'Ignoring unrecognized struct field: ''%s''.',f);
        end
      end
    end
    
  end
  
end  