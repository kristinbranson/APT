classdef ParameterVisualizationKeypointParams < ParameterVisualization
  % Embeds the LandmarkSpecs keypoint-specification UI (skeleton,
  % head/tail, swap pairs) in the visualization pane of the parameters
  % dialog.  Edits are handed back through cbkClear when the pane is
  % cleared, and are written to the Labeler only when the dialog is
  % applied.

  properties
    hPanel
    landmarkSpecs
    cbkClear
  end

  methods

    function init(obj,hTile,lObj,propFullName,prm,cbkClear,state,controller,startTabTitle)
      % Show the keypoint-specification UI in the pane holding hTile.
      % cbkClear: function handle called with the edited state when the
      %   pane is cleared.
      % state: keypoint specifications to start from, in the form
      %   Labeler.getKeypointParams() returns.
      % controller: the LabelerController, needed by LandmarkSpecs for
      %   the frozen reference image.
      init@ParameterVisualization(obj,hTile,lObj,propFullName,prm);
      obj.hPanel = hTile.Parent;
      obj.hTile.Visible = 'off';
      obj.cbkClear = cbkClear;
      if ~exist('controller','var'),
        controller = [];
      end
      if ~exist('startTabTitle','var'),
        startTabTitle = 'Swap Pairs';
      end
      obj.landmarkSpecs = LandmarkSpecs('hParent',obj.hPanel,'isVert',true,...
        'lObj',obj.lObj,'parent',controller,'state',state,...
        'startTabTitle',startTabTitle);
    end

    function s = getState(obj)
      s = obj.landmarkSpecs.getState();
    end

    function clear(obj)
      if ~isempty(obj.landmarkSpecs) && isvalid(obj.landmarkSpecs),
        s = obj.landmarkSpecs.getState();
        feval(obj.cbkClear,s);
        delete(obj.landmarkSpecs);
      end
      obj.hTile.Visible = 'on';
    end

    function update(obj) %#ok<MANU>
    end

  end

end
