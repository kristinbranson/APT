classdef LabelROI 
  methods (Static)
    function s = new()
      % s.id = ...
      s.verts = nan(4,2,0);
      s.f = nan(0,1);
    end
    % Barebones add/edit API for now. Can improve over time if nec
    function s = setF(s,v,f)
      % set/replace all rois for frame f
      %
      % v: [4 x 2 x nroi]. can be empty ([])=>just delete existing rois
      % f: [1] 
      
      % clear out existing rois for this frame
      tf = s.f==f;
      s.verts(:,:,tf) = [];
      s.f(tf,:) = [];

      if ~isempty(v)
        nroi = size(v,3);
        s.verts(:,:,end+1:end+nroi) = v;
        s.f(end+1:end+nroi,1) = f;
      end
    end
    function v = getF(s,f)
      % Get rois for frame f
      % v: [4 x 2 x nroi]
      tf = s.f==f;
      v = s.verts(:,:,tf);
    end
  end
end
