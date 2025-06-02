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
      % s = fromcoco(cocos,...)
      % Create a LabelROI structure for one movie from input cocos struct.
      % If the coco structure contains information about which movies were
      % used to create labels, then this will create a LabelROI structure
      % from data corresponding only to movie imov. 
      % Inputs:
      % cocos: struct containing data read from COCO json file
      % Fields:
      % .images: struct array with an entry for each labeled image, with
      % the following fields:
      %   .id: Unique id for this labeled image, from 0 to
      %   numel(locs.locdata)-1
      %   .file_name: Relative path to file containing this image
      %   .movid: Id of the movie this image come from, 0-indexed. This is
      %   used iff movie information is available
      %   .frmid: Index of frame this image comes from, 0-indexed. This is
      %   used iff movie information is available
      % .annotations: struct array with an entry for each annotation, with
      % the following fields:
      %   .iscrowd: Whether this is a labeled target (0) or mask (1). If
      %   not available, it is assumed that this is a labeled target (0).
      %   .segmentation: cell of length 1 containing array of length 8 x 1,
      %   containing the x (segmentation{1}(1:2:end)) and y
      %   (segmentation{1}(2:2:end)) coordinates of the mask considered
      %   labeled, 0-indexed. 
      %   .image_id: Index (0-indexed) of corresponding image
      % .info:
      %   .movies: Cell containing paths to movies. If this is available,
      %   then these movies are added to the project. 
      % Optional inputs:
      % imov: If non-empty and movie information available, only
      % information corresponding to movie imov will be used to create this
      % structure. Default: []
      % Outputs:
      % s: LabelROI structure corresponding to input data.       
      [imov] = myparse(varargin,'imov',[]);
      s = [];
      if numel(cocos.annotations) == 0 || ~isfield(cocos.annotations,'iscrowd'),
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
        imid = ann.image_id;
        imidxcurr = find([cocos.images.id]==imid);
        if hasmovies,
          s.f(i) = cocos.images(imidxcurr).frm+1; % add 1 for 1-indexing
        else
          s.f(i) = imid;
        end
      end
    end
  end
end
