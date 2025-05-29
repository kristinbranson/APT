classdef LabelROI 
  methods (Static)
    function s = new(n)
      if nargin < 1,
        n = 0;
      end
      % s.id = ...
      s.verts = nan(4,2,n);
      s.f = nan(n,1);
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
    function s = fromcoco(cocos,varargin)
      [imov] = myparse(varargin,'imov',[]);
      s = [];
      if numel(cocos.annotations) == 0 || isfield(cocos.annotations,'iscrowd'),
        return;
      end
      hasmovies = ~isempty(imov) && isfield(cocos,'info') && isfield(cocos.info,'movies');
      allnpts = [cocos.annotations.num_keypoints];
      npts = unique(allnpts(allnpts>0));
      assert(numel(npts) == 1,'All labels must have the same number of keypoints');
      if hasmovies,
        imidx = find([cocos.images.movid]==(imov-1))-1; % subtract 1 for 0-indexing
        ismov = ismember([cocos.annotations.image_id],imidx);
      else
        % assume we have created a single movie from ims, use all annotations
        ismov = true(1,numel(cocos.annotations));
      end
      islabelroi = [cocos.annotations.iscrowd]==true;
      annidx = find(ismov & islabelroi);
      n = numel(annidx);
      if n == 0,
        return;
      end
      s = LabelROI.new(n);
      for i = 1:n,
        ann = cocos.annotations(annidx(i));
        px = ann.segmentation(1:2:end);
        py = ann.segmentation(2:2:end);
        s.verts(:,1,i) = px+1; % add 1 for 1-indexing
        s.verts(:,2,i) = py+1; % add 1 for 1-indexing
        imid = ann.image_id+1; % add 1 for 1-indexing
        if hasmovies,
          s.frm(i) = cocos.images(imid).frm+1; % add 1 for 1-indexing
        else
          s.frm(i) = imid;
        end
      end
    end
  end
end
