function s = fromcoco(cocos,varargin)
  % s = fromcoco(cocos,...)
  % Create a Labels structure for one movie from input cocos struct.
  % If the coco structure contains information about which movies were
  % used to create labels, then this will create a Labels structure
  % from data corresponding only to movie imov. 
  % Input:
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
  %   .image_id: Index (0-indexed) of corresponding image
  %   .num_keypoints: Number of keypoints in this target (0 if mask)
  %   .keypoints: array of size nkeypoints*3 containing the x
  %   (keypoints(1:3:end)), y (keypoints(2:3:end)), and occlusion
  %   status (keypoints(3:3:end)). (x,y) are 0-indexed. for occlusion
  %   status, 2 means not occluded, 1 means occluded but labeled, 0
  %   means not labeled. 
  % .info:
  %   .movies: Cell containing paths to movies. If this is available,
  %   then these movies are added to the project. 
  % Optional inputs:
  % imov: If non-empty and movie information available, only
  % information corresponding to movie imov will be used to create this
  % structure. Default: []
  % tsnow: Time stamp to store in newly created labels. Default: now. 
  % Outputs:
  % s: Labels structure corresponding to input data. 
  [imov,tsnow] = myparse(varargin,'imov',[],'tsnow',now);
  s = [];
  if numel(cocos.annotations) == 0,
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
  if isfield(cocos.annotations,'iscrowd'),
    iskeypts = [cocos.annotations.iscrowd]==false;
    annidx = find(ismov & iskeypts);
  else
    annidx = find(ismov);
  end
  n = numel(annidx);
  if n == 0,
    return;
  end
  s = Labels.new(npts,n);
  s.ts(:) = tsnow;
  im2tgt = ones(1,numel(cocos.images));
  for i = 1:n,
    ann = cocos.annotations(annidx(i));
    px = ann.keypoints(1:3:end);
    py = ann.keypoints(2:3:end);
    s.p(:,i) = [px(:);py(:)]+1; % add 1 for 1-indexing
    s.occ(:,i) = 2-ann.keypoints(3:3:end);
    imid = ann.image_id; 
    imidxcurr = find([cocos.images.id]==imid);
    if hasmovies,
      s.frm(i) = cocos.images(imidxcurr).frm+1; % add 1 for 1-indexing
    else
      s.frm(i) = imid+1; % add 1 for 1-indexing
    end
    imid = ann.image_id+1; % add 1 for 1-indexing
    s.tgt(i) = im2tgt(imid);
    im2tgt(imid) = im2tgt(imid) + 1;
  end
  assert(~any(s.frm==0) && ~any(s.tgt==0));
end  % function
