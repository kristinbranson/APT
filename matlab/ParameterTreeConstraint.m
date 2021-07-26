classdef ParameterTreeConstraint < handle
  % A ParameterTreeConstraint is used at parameter-setting time to enforce
  % constraints between parameters that would be confusing for the user.
  %
  % In particular, eg a preprocessing parameter might have a constraint
  % with a CPR property; if a DL tracker is selected, the CPR properties
  % are unavailable in the Parameter pane (probably rightly so), so the
  % existing/usual mechanism of "allow the bad state to exist, but warn
  % them and don't let them Apply" won't work since they have no way to
  % alter the CPR parameter without a lot of hassle.
  %
  % ParameterTreeConstraints are applied when a property is being
  % set/changed; see propToBeSet.
  
  properties
    predicateFcn % fcn with sig tf = predicateFcn(t,propBeingSet,valBeingSet); 
        % where t is the tree rootnode(s). Return true if the actionFcn is 
        % to be called.
    actionFcn % fcn with signature:
              %   msg = actionFcn(t,propsList,propBeingSet,valBeingSet). 
              % ActionFcn is expected to update/mutate the tree in t as
              % well as the java propsList/tree in propsList
  end
  
  methods
    
    function obj = ParameterTreeConstraint(varargin)
      for i=1:2:numel(varargin)
        obj.(varargin{i}) = varargin{i+1};
      end
    end
    
    function tfPreActionPerformed = propToBeSet(obj,t,propsList,prop2bset,val2bset)
      % Apply a constraint if necessary
      % 
      % t: treenode root(s)
      % propsList: java.util.ArrayList. Must be in correspondence with t
      % prop2bset. char, FQN of property being set. eg: 
      %   'ROOT.ImageProcessing.MultiTarget.TargetCrop.AlignUsingTrxTheta'
      % val2bset. value that is being set
      %
      % At this instant, prop2bset and val2bset have NOT been set yet. They
      % will be set after constraints have had a change to perform a 
      % pre-action/mutation. The actionFcn should not set prop2bset, but
      % set other properties so that the coming set of prop2bset leads to a
      % consistent parameterset.
      %
      % ActionFcns should return a message to be shown as a warning etc.
      %
      % This treatment is going to fail with more complicated scenarios but
      % is fine for now. AL 20190127     
      
      assert(isscalar(obj));
      tfPreActionPerformed = obj.predicateFcn(t,prop2bset,val2bset);
      if tfPreActionPerformed
        msg = obj.actionFcn(t,propsList,prop2bset,val2bset); % performs pre-set mutation
        warningNoTrace(msg);
      end
    end
    
  end
  
  methods (Static)
    function obj = defaultAPTConstraints()
      % Just one for now, could turn into a vector
      obj = [ ...
        ParameterTreeConstraint(...
          'predicateFcn',@ParameterTreeConstraint.predFcnAlignTrxThetaCPRRotCorr,...
          'actionFcn',@ParameterTreeConstraint.actFcnAlignTrxThetaCPRRotCorr); ];      
%         ParameterTreeConstraint(...
%           'predicateFcn',@ParameterTreeConstraint.predFcnTargetCropRad,...
%           'actionFcn',@ParameterTreeConstraint.actFcnTargetCropRad); ...
%           ];      
    end
    function tf = predFcnAlignTrxThetaCPRRotCorr(t,p2set,v2set)
      nodeCPRRotCorr = t.findnode('CPR.RotCorrection.OrientationType');
      tf = strcmp(p2set,'MultiAnimal.TargetCrop.AlignUsingTrxTheta') && ...
        logical(v2set) && strcmp(nodeCPRRotCorr.Data.Value,'fixed');
    end
    function msg = actFcnAlignTrxThetaCPRRotCorr(t,pl,p2set,v2set)
      orientationTypeFQN = 'CPR.RotCorrection.OrientationType';
      t.setValue(orientationTypeFQN,'arbitrary');
      propjava = ParameterTreeConstraint.findGridProp(pl,orientationTypeFQN);
      propjava.setValue('arbitrary');
      msg = 'CPR OrientationType cannot be ''fixed'' if aligning target crops using trx.theta. Setting CPR OrientationType to ''arbitrary''.';
    end
    
    
    % Ugh need to modify java propsList. Details of navigation here; these
    % data structures are specific to propertiesGUI2.
    % This stuff prob should be moved back to propertiesGUI2.
    function pfound = findGridProp(propAL,propFQN)
      % propAL: java.util.ArrayList
      items = strread(propFQN,'%s','delimiter','.');
      pfound = ParameterTreeConstraint.findGridPropRecurse(propAL,items);
    end
    function pfound = findGridPropRecurse(propAL,items)
      n = propAL.size;
      for i=0:n-1 % java
        p = propAL.get(i);
        if strcmp(ParameterTreeConstraint.getPropName(p),items{1})
          if isscalar(items) % found it
            pfound = p;
          else
            pfound = ParameterTreeConstraint.findGridPropRecurse(p.getChildren,items(2:end));
          end
        end
      end
    end
    % C+P from propertiesGUI2
    function propName = getPropName(hProp)
      try
        propName = get(hProp,'UserData');
      catch
        %propName = char(getappdata(hProp,'UserData'));
        propName = get(handle(hProp),'UserData');
      end
    end 
  end
  
end
  
  