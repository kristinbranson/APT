classdef TreeNode < handle 
  % Seems to be a general handle class for representing a rooted tree, where
  % each node can have some data associated with it.  -- ALT, 2025-02-04

  properties
    Data
    Children % vector of TreeNodes
  end
  
  methods
    
    function obj = TreeNode(dat,isjsonstruct,varargin)
      if nargin >= 2 && isjsonstruct,
        obj.initFromJsonStruct(dat,varargin{:});
      else
        obj.Children = TreeNode.empty(0,1);
        obj.Data = dat;
      end
    end
    
    function traverse(t,fcn)
      assert(isscalar(t));
      fcn(t);
      c = t.Children;
      for i=1:numel(c)
        c(i).traverse(fcn);
      end
    end
    
    function traverseDispWithPath(t,fcn)
      % fcn should return a string to display, given the .Data from a single node.
      
      assert(isscalar(t));
      trav(t,'');
      function trav(zt,zpth)
        str = fcn(zt.Data);
        zpthnew = [zpth '.' zt.Data.Field];
        fprintf(1,'%s: %s\n',zpthnew,str);
        c = zt.Children;
        for i=1:numel(c)
          trav(c(i),zpthnew);
        end
      end
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
    
    function s = jsonify(t)
      % Convert Tree to a struct for outputting to json file
      s = struct;
      isleaf = numel(t.Children) == 0;
      fns = setdiff(fieldnames(t.Data),{'Field','FullPath','Index','UserData','Visible','DispNameUse','Value'});
      for i = 1:numel(fns),
        fn = fns{i};
        if ~isleaf && ismember(fn,{'Type','ParamViz','Level','AffectsTraining','Requirements','DefaultValue','Value','isEditable'}),
          continue;
        end
        s.(fn) = t.Data.(fn);
      end
      for i = 1:numel(t.Children),
        fld = t.Children(i).Data.Field;
        s.(fld) = jsonify(t.Children(i));
      end
    end

    function initFromJsonStruct(obj,s,varargin)
      [fld,fcnStruct2Data,prefix] = myparse(varargin,'Field','ROOT',...
        'fcnStruct2Data',@PropertiesGUIProp.initFromStruct,'prefix','');
      [obj.Data,fnsused] = fcnStruct2Data(s,fld,prefix);
      childrenflds = setdiff(fieldnames(s),fnsused);
      obj.Children = TreeNode.empty(0,1);
      if isempty(prefix),
        prefix = fld;
      else
        prefix = [prefix,'.',fld];
      end
      for i = 1:numel(childrenflds),
        childfld = childrenflds{i};
        obj.Children(i,1) = TreeNode(s.(childfld),true,'Field',childfld,'fcnStruct2Data',fcnStruct2Data,'prefix',prefix);
      end
    end

    function tcopy = copy(t)
      % Deep copy

      % Atm we only support the PropertiesGUIProp type in .Data, as this
      % class guarantees a full deep copy(). Any object which is 
      % fully/deep-copied with copy() should be fine though.
      assert(isa(t.Data,'PropertiesGUIProp'));
      pgp = t.Data.copy();
      tcopy = TreeNode(pgp);
      for i = 1:numel(t.Children),
        tcopy.Children(i) = t.Children(i).copy();
      end
      tcopy.Children = reshape(tcopy.Children,size(t.Children));
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
          elseif isempty(val),
            % MK 20190110, val can be an empty array
            if isempty(node.Children),
              node.Data.Value = val;
            end
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
    
    function n = findnode(t,propFQN)
      % find node with fully qualified name 
      %
      % t: root nodes(s)
      % propFQN: "fully qualified name", 
      %     eg 'ROOT.ImageProcessing.MultiTarget.TargetCrop.AlignUsingTrxTheta'
      %
      % n: scalar PropertiesGUIProp node, or [] if not found.
      
      items = strread(propFQN,'%s','delimiter','.');

      for i=1:numel(items)
        it = items{i};
        flds = arrayfun(@(x)x.Data.Field,t,'uni',0);
        tf = strcmp(flds,it);
        if nnz(tf)~=1
          n = [];
          return;
        end
        n = t(tf);
        t = n.Children;
      end
    end

    function [n,p] = findfield(t,fn,leafonly)
      if nargin < 3,
        leafonly = false;
      end
      n = [];
      p = {};
      if strcmp(t.Data.Field,fn) && (~leafonly || numel(t.Children) == 0),
        n = t;
        p = {[]};
      end
      for c = t.Children(:)',
        [nc,pc] = findfield(c,fn,leafonly);
        if isempty(nc),
          continue;
        end
        for i = 1:numel(pc),
          if isempty(pc{i}),
            pc{i} = t;
          else
            pc{i} = [t,pc{i}];
          end
        end
        if isempty(n),
          n = nc;
          p = pc;
        else
          n(end+1:end+numel(nc)) = nc;
          p(end+1:end+numel(nc)) = pc;
        end
      end
    end
    
    function nodelist = flatten(t)
      % return flat nodelist
      
      nodelist = cell(0,1);      
      t.traverse(@lclAccum);
      
      nodelist = cat(1,nodelist{:});
      
      function lclAccum(x)
        nodelist{end+1,1} = x;
      end
    end
    
    function nrepeatNodes = countRepeatNodes(t)
      % nrepeatNodes: integer. if nodelist=t.flatten(), number of nodes
      % in nodelist that appear at least twice in nodelist.
      
      nodelist = t.flatten();
      cnt = arrayfun(@(x)nnz(nodelist==x),nodelist);
      nrepeatNodes = nnz(cnt>1);      
    end
    
    function setValue(t,propFQN,propVal)
      % Set the property specified by propFQN to have the value propVal.
      %
      % t: root nodes(s)
      % propFQN: "fully qualified name", 
      %     eg 'ROOT.ImageProcessing.MultiTarget.TargetCrop.AlignUsingTrxTheta'
      % propValue: value to be set into Value field
      
      node = t.findnode(propFQN);
      if isempty(node)
        error('Could not find node %s.',propFQN);
      end
      node.Data.Value = propVal;      
    end
    
    function v = getValue(t,propFQN)
      node = t.findnode(propFQN);
      if isempty(node)
        error('Could not find node %s.',propFQN);
      end
      v = node.Data.Value;
    end
        
  end
  
end
