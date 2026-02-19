function cocos = ConvertTrnPackLocToCOCO(locs,packdir,varargin)
  % cocos = ConvertTrnPackLocToCOCO(locs,packdir,...)
  % Convert from the original format for json data to the COCO format
  % Inputs:
  % locs: struct that is created as part of genWriteTrnPack()
  %   Required to have the following fields:
  %   movies: cell of size nmovies x 1 containing file paths to movies
  %   locdata: struct array with an entry for each labeled image:
  %     .img: relative path to image created in the cache directory
  %     .imov: index of movie for this image
  %     .frm: frame number
  %     .ntgt: number of labeled targets
  %     .roi: array of size 8 x ntgt, containing the x (roi(1:4)) and y
  %     (roi(5:8)) coordinates of the corners of the box considered
  %     labeled around this target
  %     .pabs: array of size (nkeypoints*2) x ntgt, x (1:nkeypoints)
  %     and y (nkeypoints+1:end) coordinates of keypoints.
  %     .occ: array of size nkeypoints x ntgt, 0 = not occluded, 1 =
  %     occluded but labeled, 2 = not labeled
  %     .nextra_roi: number of extra boxes marked as labeled
  %     .extra_roi: 8 x nextra_roi, containing the x (extra_roi(1:4))
  %     and y (extra_roi(5:8)) coordinates of the corners of the box
  %     considered labeled
  % packdir: Path to root directory to images
  % Optional inputs:
  % imwidth: Width of all images. Otherwise, will be read from images
  % imheight: Height of all images. Otherwise, will be read from images
  % skeleton: Array of size nedges x 2 indicating which keypoints
  % should be connected by a skeleton. If empty, it will just be
  % [(1:nkeypoints-1)',(2:nkeypoints)']
  % animaltype: Name for this type of animal. Default: 'animal'
  % keypoint_names: Names for each keypoint. Only added to cocos.info
  % if not empty. Default: {}
  % Outputs:
  % cocos: Struct containing data reformated so that it can be saved to
  % COCO file format with jsonencode. Fields:
  % .images: struct array with an entry for each labeled image, with
  % the following fields:
  %   .id: Unique id for this labeled image, from 0 to
  %   numel(locs.locdata)-1
  %   .width, .height: Width and height of the image
  %   .file_name: Relative path to file containing this image
  %   .moivid: Id of the movie this image come from, 0-indexed
  %   .frmid: Index of frame this image comes from, 0-indexed
  %   .patch: Not sure what this is, set it to ntgt+nextra-1
  % .annotations: struct array with an entry for each annotation, with
  % the following fields:
  %   .iscrowd: Whether this is a labeled target (0) or mask (1)
  %   .segmentation: cell of length 1 containing array of length 8 x 1,
  %   containing the x (segmentation{1}(1:2:end)) and y
  %   (segmentation{1}(2:2:end)) coordinates of the mask considered
  %   labeled, 0-indexed
  %   .area: Area of annotation. If this is a target, then it is the
  %   area of the tight bounding box. If it is a mask, area of the
  %   mask.
  %   .image_id: Index (0-indexed) of corresponding image
  %   .num_keypoints: Number of keypoints in this target (0 if mask)
  %   .bbox: Tight bounding box around keypoints if target, same as
  %   segmentation if mask. Array of size 1x8 containing the x
  %   (bbox(1:2:end)) and y (bbox(2:2:end)) coordinates of the corner
  %   of the box. 0-indexed
  %   .keypoints: array of size nkeypoints*3 containing the x
  %   (keypoints(1:3:end)), y (keypoints(2:3:end)), and occlusion
  %   status (keypoints(3:3:end)). (x,y) are 0-indexed. for occlusion
  %   status, 2 means not occluded, 1 means occluded but labeled, 0
  %   means not labeled.
  %   .category_id: 1 if a target, 2 if a label mas box
  [imwidth,imheight,skeleton,animaltype,keypoint_names,isma] = myparse(varargin,'imwidth',[],'imheight',[],'skeleton',[],'animaltype','animal','keypoint_names',{},'isma',true);

  isimsize = ~isempty(imwidth) && ~isempty(imheight);

  cocos = struct;
  imagetemplate = struct('id', 0, ...
    'width', 0, 'height', 0,...
    'file_name', '',...
    'movid', 0, ...
    'frm', 0,...
    'patch',0);
  nims = numel(locs.locdata);
  anntemplate = struct('iscrowd',false,'segmentation',[],'area',0,'image_id',0,'id',0,'num_keypoints',0,'bbox',[],'keypoints',[],'category_id',0);
  cocos.images = repmat(imagetemplate,[nims,1]);
  nann = sum([locs.locdata.ntgt]);
  if isfield(locs.locdata,'nextra_roi'),
    nann = nann + sum([locs.locdata.nextra_roi]);
  end
  cocos.annotations = repmat(anntemplate,[nann,1]);

  annid = 0;
  nkeypoints = 0;
  for i = 1:nims,
    loccurr = locs.locdata(i);
    imcurr = imagetemplate;
    imcurr.id = i-1;
    imcurr.file_name = loccurr.img{1};
    if isimsize,
      imcurr.width = imwidth;
      imcurr.height = imheight;
    else
      fp = fullfile(packdir,imcurr.file_name);
      info = imfinfo(fp);
      imcurr.width = info.Width;
      imcurr.height = info.Height;
    end
    imcurr.movid = loccurr.imov-1;
    imcurr.frm = loccurr.frm-1;
    imcurr.patch = loccurr.ntgt;
    if isfield(loccurr,'nextra_roi'),
      imcurr.patch = imcurr.patch + loccurr.nextra_roi - 1; % patch is the number of targets + number of extra rois - 1 for some reason
    end
    cocos.images(i) = imcurr;
    for j = 1:loccurr.ntgt,
      anncurr = anntemplate;
      anncurr.iscrowd = false;
      segx = loccurr.roi(1:size(loccurr.roi,1)/2,j)-1;
      segy = loccurr.roi(size(loccurr.roi,1)/2+1:end,j)-1;
      anncurr.segmentation = [segx(:),segy(:)]';
      anncurr.segmentation = {anncurr.segmentation(:)'};
      % seems like whether this is stored as a row or a column
      % varies...
      if isma,
        p = loccurr.pabs;
        occ = loccurr.occ;
      else
        p = loccurr.pabs';
        occ = loccurr.occ';
      end
      px = p(1:size(p,1)/2,j)-1;
      py = p(size(p,1)/2+1:end,j)-1;
      occ = 2-double(occ(:,j));
      minx = min(px);
      maxx = max(px);
      miny = min(py);
      maxy = max(py);
      anncurr.area = (maxy-miny)*(maxx-minx);
      anncurr.image_id = i-1;
      anncurr.id = annid;
      anncurr.num_keypoints = numel(px);
      nkeypoints = numel(px);
      anncurr.bbox = [minx,miny,maxx-minx,maxy-miny];
      anncurr.keypoints = [px(:),py(:),occ(:)]';
      anncurr.keypoints = anncurr.keypoints(:);
      anncurr.category_id = 1;
      annid = annid + 1;
      cocos.annotations(annid) = anncurr;
    end
    if isfield(loccurr,'nextra_roi'),
      for j = 1:loccurr.nextra_roi,
        anncurr = anntemplate;
        anncurr.iscrowd = true;
        roi = loccurr.extra_roi(:,j);
        segx = roi(1:size(roi,1)/2)-1;
        segy = roi(size(roi,1)/2+1:end)-1;
        anncurr.segmentation = [segx(:),segy(:)]';
        anncurr.segmentation = {anncurr.segmentation(:)'};
        minx = min(segx);
        maxx = max(segx);
        miny = min(segy);
        maxy = max(segy);
        anncurr.area = (maxy-miny)*(maxx-minx);
        anncurr.image_id = i-1;
        anncurr.id = annid;
        anncurr.num_keypoints = 0;
        anncurr.bbox = [];
        anncurr.keypoints = [];
        anncurr.category_id = 2;
        annid = annid + 1;
        cocos.annotations(annid) = anncurr;
      end
    end
  end

  if numel(cocos.images) <= 1,
    cocos.images = {cocos.images};
  end
  if numel(cocos.annotations) <= 1,
    cocos.annotations = {cocos.annotations};
  end

  cocos.info = struct;
  cocos.info.movies = locs.movies;
  if ~isempty(keypoint_names),
    cocos.info.keypoint_names = keypoint_names;
  end
  if isempty(skeleton),
    skeleton = [(0:nkeypoints-1)',(1:nkeypoints)'];
  end
  if size(skeleton,1) == 1 || size(skeleton,2) == 1,
    skeleton = {{skeleton}};
  end
  catkpt = struct('id',1,'skeleton',skeleton,'super_category',animaltype,'name',animaltype);
  catmask = struct('id',2,'skeleton',[],'super_category','mask_box','name','mask_box');
  cocos.categories = [catkpt;catmask];
end % function
