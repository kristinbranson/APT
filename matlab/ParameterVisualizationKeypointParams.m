classdef ParameterVisualizationKeypointParams < ParameterVisualization
  
  properties
    hPanel
    landmarkSpecs
    cbkClear
  end
  
  methods
        
    function init(obj,hTile,lObj,propFullName,prm,cbkClear,state,startTabTitle)
      
      init@ParameterVisualization(obj,hTile,lObj,propFullName,prm);
      obj.hPanel = hTile.Parent;
      obj.hTile.Visible = 'off';
      obj.cbkClear = cbkClear;
      if ~exist('startTabTitle','var'),
        startTabTitle = 'Correspondences';
      end
      obj.landmarkSpecs = landmark_specs('lObj',obj.lObj,'startTabTitle',startTabTitle,'hParent',obj.hPanel,'state',state,'isVert',true);

    end

    function s = getState(obj)
      s = obj.landmarkSpecs.getState();
    end

    function clear(obj)
      if ishandle(obj.landmarkSpecs),
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