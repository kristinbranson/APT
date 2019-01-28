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
              %   msg = actionFcn(t,propBeingSet,valBeingSet). 
              % ActionFcn is expected to update/mutate the tree.
  end
  
  methods
    
    function obj = ParameterTreeConstraint(varargin)
      for i=1:2:numel(varargin)
        obj.(varargin{i}) = varargin{i+1};
      end
    end
    
    function tfPreActionPerformed = propToBeSet(obj,t,prop2bset,val2bset)
      % Apply a constraint if necessary
      % 
      % t: treenode root(s)
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
        msg = obj.actionFcn(t,prop2bset,val2bset); % performs pre-set mutation
        warningNoTrace(msg);
      end
    end
    
  end
  
  methods (Static)
    function obj = defaultAPTConstraints()
      % Just one for now, could turn into a vector
      obj = ParameterTreeConstraint(...
        'predicateFcn',@ParameterTreeConstraint.predFcnAlignTrxThetaCPRRotCorr,...
        'actionFcn',@ParameterTreeConstraint.actFcnAlignTrxThetaCPRRotCorr);
    end
    function tf = predFcnAlignTrxThetaCPRRotCorr(t,p2set,v2set)
      nodeCPRRotCorr = t.findnode('CPR.RotCorrection.OrientationType');
      tf = strcmp(p2set,'ImageProcessing.MultiTarget.TargetCrop.AlignUsingTrxTheta') && ...
        logical(v2set) && strcmp(nodeCPRRotCorr.Data.Value,'fixed');
    end
    function msg = actFcnAlignTrxThetaCPRRotCorr(t,p2set,v2set)
      t.setValue('CPR.RotCorrection.OrientationType','arbitrary');
      msg = 'CPR OrientationType cannot be ''fixed'' if aligning target crops using trx.theta. Setting CPR OrientationType to ''arbitrary''.';
    end
  end
  
end
  
  