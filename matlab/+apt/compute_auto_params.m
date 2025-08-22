function [autoparams,vizdata] = compute_auto_params(lobj,varargin)

  [prmtree,SCALE_PRCTILE_SPAN,...
    THRESH_MULTI_BBOX_SCALE,...
    CROP_RADIUS_PRECISION,...
    TRANSLATION_PRCTILE_D_PAIRS,...
    TRANSLATION_FRAC_FRAMESIZE,...
    TRANSLATION_FRAC_SPAN,...
    THRESH_DPAIR_VS_SPAN,...
    ROTATION_RANGE_ALIGN_THETA,...
    TRANSLATION_RANGE_PRECISION,...
    ROTATION_RANGE_PRECISION,...
    ROTATION_PRCTILE_ANGLE_SPAN] = myparse(varargin,...
    'prmtree',[],'scale_prctile_span',5,...
    'thresh_multi_bbox_scale',2,...
    'crop_radius_precision',16,...
    'translation_prctile_d_pairs',5,...
    'translation_frac_framesize',10,...
    'translation_frac_span',10,...
    'thresh_dpair_vs_span',10,...
    'rotation_range_align_theta',15,...  % degrees
    'translation_range_precision',5,...
    'rotation_range_precision',10,...
    'rotation_range_angle_span',2);

  assert(lobj.nview==1, 'Auto setting of parameters not tested for multivew');

  if isempty(prmtree),
    sPrmCurrent = lobj.trackGetTrainingParams();
    prmtree = APTParameters.defaultParamsTree() ;
    % Overlay our starting point
    prmtree.structapply(sPrmCurrent);
  end

  %% collate all labels and compute distances between centroids of bounding boxes of pairs of animals labeled on the same frame
  view = 1;
  autoparams = containers.Map();
  vizdata = struct;

  nmov = numel(lobj.labels);
  npts = lobj.nLabelPoints;
  d = 2;
  cur_pts = ((view-1)*2*npts+1):(view*d*npts);
  all_labels = zeros(numel(cur_pts),0);
  all_id = [];
  all_mov = [];
  d_pairs = [];
  for movi = 1:nmov        
    if ~Labels.hasLbls(lobj.labels{movi}), continue; end

    n_labels = size(lobj.labels{movi}.p,2);
    all_labels(:,end+1:end+n_labels,:) = lobj.labels{movi}.p(cur_pts,:);
    all_id(end+1:end+n_labels) = 1:n_labels;
    all_mov(end+1:end+n_labels) = movi;

    if ~lobj.maIsMA && ~lobj.hasTrx, continue, end
    % loop through unique frames
    uniquefrms = unique(lobj.labels{movi}.frm);
    for fi = 1:numel(uniquefrms),
      f = uniquefrms(fi);
      idx = find(lobj.labels{movi}.frm==f);
      % loop through unique pairs of labels for current frame
      for ii = 1:numel(idx)-1,
        i = idx(ii);
        xy1 = reshape(lobj.labels{movi}.p(cur_pts,i),[npts,d]);
        % compute bounding box center
        minxy = min(xy1,[],1);
        maxxy = max(xy1,[],1);
        ctrxy1 = (minxy+maxxy)/2; % [1,2]
        for jj = ii+1:numel(idx),
          j = idx(jj);
          xy2 = reshape(lobj.labels{movi}.p(cur_pts,j),[npts,d]);
          % compute bounding box center
          minxy = min(xy2,[],1);
          maxxy = max(xy2,[],1);
          ctrxy2 = (minxy+maxxy)/2; % [1,2]

          % distance between bounding box centers
          d12 = sqrt(sum((ctrxy1 - ctrxy2).^2)); % scalar

          d_pairs(end+1) = d12; %#ok<AGROW>
        end
      end
    end
  end

  all_labels = reshape(all_labels,npts,d,[]); % all_labels is npts x d x nlabels
  fprintf('New: numel(d_pairs) = %d\n',numel(d_pairs));

  %% scaling-related parameters

  % l_min and l_max are corners of the bounding box of each label
  l_min = permute(min(all_labels,[],1,'omitnan'),[2,3,1]); 
  l_max = permute(max(all_labels,[],1,'omitnan'),[2,3,1]); 
  l_span = l_max-l_min; % [d,nlabels] 

  % KB 20250717 switched this to using the the diagonal
  l_diagonal = sqrt(sum((l_min - l_max).^2,1));

  vizdata.scale = struct;
  vizdata.scale.bboxDiagonalLength = l_diagonal;
  vizdata.scale.bboxWidthHeight = l_span;

  l_diagonal_pc = prctile(l_diagonal,[50,100-SCALE_PRCTILE_SPAN],2);
  big_l_diagonal = l_diagonal_pc(2);
  med_l_diagonal = l_diagonal_pc(1);

  vizdata.scale.bigBboxDiagonalLength  = big_l_diagonal;
  vizdata.scale.medBboxDiagonalLength  = med_l_diagonal;

  % switched the order of operations: take the max over x and y spans (ignore whether 
  % this a horizontal box or a vertical box) and then take the percentiles
  % to me to take the max first and then take the percentile
  auto_multi_bbox_scale = big_l_diagonal/med_l_diagonal > THRESH_MULTI_BBOX_SCALE;
  % set whether to crop detections by their detected size based on the
  % variations observed in the labels

  if lobj.trackerIsTwoStage,
    autoparams('ROOT.MultiAnimal.TargetCrop.multi_scale_by_bbox') = auto_multi_bbox_scale;
  end

    %l_min = reshape(min(all_labels,[],1),size(all_labels,[2,3]));
    %l_max = reshape(max(all_labels,[],1),size(all_labels,[2,3]));
    % l_span is labels span in x and y direction
  
  l_span_pc = prctile(l_span,100-SCALE_PRCTILE_SPAN,2); % [w;h]
  l_span_max = max(l_span,[],2); % box size
  vizdata.scale.bigBboxSide = l_span_max;

  % Check and flag outliers..
  if any( (l_span_max./l_span_pc)>2)
    outliers = zeros(0,3);
    for dim = 1:2, % x or y
      ix = find(l_span(dim,:)>l_span_pc(dim)*THRESH_MULTI_BBOX_SCALE); 
      for xx = ix(:)'
        mov = all_mov(xx);
        yy = all_id(xx);
        cur_fr = lobj.labels{mov}.frm(yy);
        cur_tgt = lobj.labels{mov}.tgt(yy);
        outliers(end+1,:) = [mov,cur_fr,cur_tgt]; %#ok<AGROW>
      end
    end
    wstr = cell(1,size(outliers,1)+2);
    wstr{1} = sprintf('Some bounding boxes have sizes larger than %d times the %dth percentile. These could be labeling errors. ',THRESH_MULTI_BBOX_SCALE,100-SCALE_PRCTILE_SPAN);
    wstr{2} = 'The list of examples is:';
    for zz = 1:size(outliers,1)
      wstr{zz+2} = sprintf('Movie:%d, frame:%d, target:%d\n',outliers(zz,1),outliers(zz,2),outliers(zz,3));
    end
    warningNoTrace(sprintf('%s\n',wstr{:}));
  end
  l_span_median = prctile(l_span,50,2); % [w;h]
  vizdata.scale.medBboxWidthHeight = l_span_median;

  if lobj.trackerIsTwoStage && auto_multi_bbox_scale,
    % scaling bounding box == true, size to scale to

    % median span in x and y 
    crop_radius = max(l_span_median);
    
  else
    crop_radius = max(l_span_pc);
  end
  crop_radius = ceil(crop_radius/CROP_RADIUS_PRECISION)*CROP_RADIUS_PRECISION;

  autoparams('ROOT.MultiAnimal.TargetCrop.ManualRadius') = crop_radius;


  %% rotation range parameters

  if ~lobj.trackerIsTwoStage && ~lobj.hasTrx
    autoparams('ROOT.MultiAnimal.TargetCrop.AlignUsingTrxTheta') = false;
  end

  laststage_horzflip = prmtree.findnode([APTParameters.poseDataAugPath,'.horz_flip']).Data.Value;
  laststage_vertflip = prmtree.findnode([APTParameters.poseDataAugPath,'.vert_flip']).Data.Value;

  firststage_horzflip = prmtree.findnode([APTParameters.detectDataAugPath,'.horz_flip']).Data.Value;
  firststage_vertflip = prmtree.findnode([APTParameters.detectDataAugPath,'.vert_flip']).Data.Value;

  % flip to best match template
  [all_labels_horzflipped,all_labels_vertflipped] = flip_labels(all_labels,lobj);

  vizdata.rrange = struct;

  if lobj.maIsMA && lobj.trackerIsTwoStage,

    if lobj.trackParams.ROOT.MultiAnimal.TargetCrop.AlignUsingTrxTheta,
      % compute span of angles head and tail keypoints around midpoint of
      % head and tail.
      headidx = lobj.skelHead;
      tailidx = lobj.skelTail;
      if isempty(headidx),
        warningNoTrace('Head and tail keypoint indices MUST be set if aligning with body axis.');
        headidx = 1;
        tailidx = 2;
      end

      if firststage_horzflip,
        all_labels_use = all_labels_horzflipped;
      elseif firststage_vertflip,
        all_labels_use = all_labels_vertflipped;
      else
        all_labels_use = all_labels;
      end

      [rrange_stage1,l_thetas,minthetao] = rrange_headtail_around_centroid(all_labels_use,headidx,tailidx,ROTATION_PRCTILE_ANGLE_SPAN,ROTATION_RANGE_PRECISION);
      vizdata.rrange.firststage_headTailAngle = l_thetas;
      vizdata.rrange.offset.firststage_headTailAngle = minthetao;

      % we have already cropped a region around the animal and aligned
      % based on detected head/tail position. compute span of difference
      % between keypoint angles and head-tail angle
      if laststage_horzflip,
        all_labels_use = all_labels_horzflipped;
      elseif laststage_vertflip,
        all_labels_use = all_labels_vertflipped;
      else
        all_labels_use = all_labels;
      end
      [rrange_stage2,l_thetas,minthetao] = rrange_keypoints_relative_headtail(all_labels_use,headidx,tailidx,ROTATION_PRCTILE_ANGLE_SPAN,ROTATION_RANGE_ALIGN_THETA,ROTATION_RANGE_PRECISION);
      vizdata.rrange.laststage_keypoints2HeadTailAngle = l_thetas;
      vizdata.rrange.offset.laststage_keypoints2HeadTailAngle = minthetao;
    else

      if firststage_horzflip,
        all_labels_use = all_labels_horzflipped;
      elseif firststage_vertflip,
        all_labels_use = all_labels_vertflipped;
      else
        all_labels_use = all_labels;
      end

      [rrange_stage1,l_thetas,minthetao] = rrange_keypoints_around_centroid(all_labels_use,ROTATION_PRCTILE_ANGLE_SPAN,ROTATION_RANGE_PRECISION);
      vizdata.rrange.firststage_centroidKeypointAngle = l_thetas;
      vizdata.rrange.offset.firststage_centroidKeypointAngle = minthetao;

      if laststage_horzflip,
        all_labels_use = all_labels_horzflipped;
      elseif laststage_vertflip,
        all_labels_use = all_labels_vertflipped;
      else
        all_labels_use = all_labels;
      end

      [rrange_stage2,l_thetas,minthetao] = rrange_keypoints_around_centroid(all_labels_use,ROTATION_PRCTILE_ANGLE_SPAN,ROTATION_RANGE_PRECISION);
      vizdata.rrange.laststage_centroidKeypointAngle = l_thetas;
      vizdata.rrange.offset.laststage_centroidKeypointAngle = minthetao;

    end

    autoparams([APTParameters.detectDataAugPath,'.rrange']) = rrange_stage1;
    autoparams([APTParameters.poseDataAugPath,'.rrange']) = rrange_stage2;
  else

    if ~lobj.maIsMA && lobj.hasTrx && lobj.trackParams.ROOT.MultiAnimal.TargetCrop.AlignUsingTrxTheta,
      % we have already cropped a region around the animal using hasTrx and
      % have aligned with trx angle. we don't have the trx info cached, so
      % let's just set this to a constant
      rrange = ROTATION_RANGE_ALIGN_THETA;
    else
      if laststage_horzflip,
        all_labels_use = all_labels_horzflipped;
      elseif laststage_vertflip,
        all_labels_use = all_labels_vertflipped;
      else
        all_labels_use = all_labels;
      end

      [rrange,l_thetas,minthetao] = rrange_keypoints_around_centroid(all_labels_use,ROTATION_PRCTILE_ANGLE_SPAN,ROTATION_RANGE_PRECISION);
      vizdata.rrange.centroidKeypointAngle = l_thetas;
      vizdata.rrange.offset.centroidKeypointAngle = minthetao;
    end
    autoparams([APTParameters.poseDataAugPath,'.rrange']) = rrange;
  end

  %% translation-related parameters

  % Look at distances between labeled pairs to find what to set for
  % translation range for data augmentation
  d_pairs_pc = prctile(d_pairs,TRANSLATION_PRCTILE_D_PAIRS);

  vizdata.trange = struct;
  vizdata.trange.distPairs = d_pairs;
  vizdata.trange.smallDistPairs = d_pairs_pc;

  % If the distances between center of animals is much less than the
  % span then warn about using bbox based methods
  if(d_pairs_pc<min(l_span_pc/THRESH_DPAIR_VS_SPAN)) && lobj.trackerIsObjDet
    wstr = {'The distances between the center of animals is much smaller than the size of the animals';
      'Avoid using object detection based top-down methods'};
    warning(sprintf('%s\n',wstr{:}));  %#ok<SPWRN>
  end

  % translation range based on frame size
  trange_frame = round_nearest(min(lobj.movienr(view),lobj.movienc(view))/TRANSLATION_FRAC_FRAMESIZE,TRANSLATION_RANGE_PRECISION);

  % translation range based on distance between animals
  trange_pair = round_nearest(d_pairs_pc/2,TRANSLATION_RANGE_PRECISION);

  % translation range based on bounding box
  trange_crop = round_nearest(min(l_span_pc)/TRANSLATION_FRAC_SPAN,TRANSLATION_RANGE_PRECISION);

  fprintf('New: trange_frame: %.1f, trange_pair: %.1f, trange_crop: %.1f\n',...
    trange_frame,trange_pair,trange_crop);

  if lobj.maIsMA && lobj.trackParams.ROOT.MultiAnimal.multi_crop_ims,
    % If we are cutting the images into pieces (i think this is only for
    % bottom up?), use this size instead of frame size
    crop_sz = lobj.trackParams.ROOT.MultiAnimal.multi_crop_im_sz;
    trange_frame = round_nearest(crop_sz/TRANSLATION_FRAC_SPAN,TRANSLATION_RANGE_PRECISION);
  end

  if lobj.maIsMA && lobj.trackerIsTwoStage,

    trange_stage1 = trange_frame;
    trange_stage2 = min([trange_crop,trange_frame,trange_pair]);

    autoparams([APTParameters.detectDataAugPath,'.trange']) = trange_stage1;
    autoparams([APTParameters.poseDataAugPath,'.trange']) = trange_stage2;

  else
    
    if ~lobj.maIsMA && lobj.hasTrx,
      trange = min([trange_crop,trange_frame,trange_pair]);
    else
      trange = trange_frame;
    end

    autoparams([APTParameters.poseDataAugPath,'.trange']) = trange;

  end

end

%% helper functions

function [ang_span,med_theta] = get_angle_span(theta,ROTATION_PRCTILE_ANGLE_SPAN)
  % Find the span of thetas. Hacky method that rotates the pts by
  % 10 degrees and then checks the span.
  ang_span = ones(size(theta,1),1)*2*pi;
  med_theta = nan(size(theta,1),1);
  for offset = -180:10:180,
    thetao = modrange(theta+offset*pi/180,-pi,pi);
    theta1 = prctile(thetao,100-ROTATION_PRCTILE_ANGLE_SPAN,3);
    theta0 = prctile(thetao,ROTATION_PRCTILE_ANGLE_SPAN,3);
    cur_span = theta1 - theta0; % [npts,1]t
    isbest = cur_span <= ang_span;
    ang_span(isbest) = cur_span(isbest);
    % med_theta_curr = modrange((theta0+theta1)/2 - offset*pi/180,-pi,pi);
    % med_theta(isbest) = med_theta_curr(isbest);

    for i = find(isbest(:)'),
      med_theta_curr = median(thetao(i,1,thetao(i,1,:)>=theta0(i) & thetao(i,1,:)<=theta1(i)));
      med_theta_curr = modrange(med_theta_curr-offset*pi/180,-pi,pi);
      med_theta(i) = med_theta_curr;
    end

  end
end

function [rrange,l_thetas,minthetao] = rrange_headtail_around_centroid(all_labels,headidx,tailidx,ROTATION_PRCTILE_ANGLE_SPAN,ROTATION_RANGE_PRECISION)

htidx = [headidx,tailidx];
mid_labels = mean(all_labels,1); % should this be mean(all_labels(htidx)) ?
[rrange,l_thetas,minthetao] = helper_rotation_range(all_labels(htidx,:,:),mid_labels,ROTATION_PRCTILE_ANGLE_SPAN,ROTATION_RANGE_PRECISION);

end

function [rrange,l_thetas,minthetao] = rrange_keypoints_relative_headtail(all_labels,headidx,tailidx,ROTATION_PRCTILE_ANGLE_SPAN,ROTATION_RANGE_ALIGN_THETA,ROTATION_RANGE_PRECISION)

npts = size(all_labels,1);
if npts <= 2,
  rrange = ROTATION_RANGE_ALIGN_THETA;
  return;
end
ptidx = true(1,npts);
ptidx([headidx,tailidx]) = false;

hd = all_labels(headidx,:,:);
tl = all_labels(tailidx,:,:);
body_ctr = (hd+tl)/2;

htangle = atan2(all_labels(headidx,2,:)-body_ctr(:,2,:),...
  all_labels(headidx,1,:)-body_ctr(:,1,:));

[rrange,l_thetas,minthetao] = helper_rotation_range(all_labels(ptidx,:,:),body_ctr,ROTATION_PRCTILE_ANGLE_SPAN,ROTATION_RANGE_PRECISION,htangle);

end

function [rrange,l_thetas,minthetao] = rrange_keypoints_around_centroid(all_labels,ROTATION_PRCTILE_ANGLE_SPAN,ROTATION_RANGE_PRECISION)

% If uses object detection for the first stage or hastrx and not 
% using theta to align 
% then the look at the variation in angles relative to the center of the
% keypoints
mid_labels = mean(all_labels,1);
[rrange,l_thetas,minthetao] = helper_rotation_range(all_labels,mid_labels,ROTATION_PRCTILE_ANGLE_SPAN,ROTATION_RANGE_PRECISION);

end

function [rrange,l_thetas,minthetao] = helper_rotation_range(all_labels,ctrpts,ROTATION_PRCTILE_ANGLE_SPAN,ROTATION_RANGE_PRECISION,ctrangles)

assert(numel(ROTATION_PRCTILE_ANGLE_SPAN) == 1);
assert(numel(ROTATION_RANGE_PRECISION) == 1);

% KB: this had dx and dy in the wrong order, as far as i understand what it's
% doing. hopefully doesn't matter
l_thetas = atan2(all_labels(:,2,:)-ctrpts(:,2,:),...
  all_labels(:,1,:)-ctrpts(:,1,:));

if exist('ctrangles','var'),
  l_thetas = modrange(l_thetas - ctrangles,-pi,pi);
end

[ang_span,minthetao] = get_angle_span(l_thetas,ROTATION_PRCTILE_ANGLE_SPAN);
ang_span = ang_span*180/pi;
rrange = median(ang_span)/2;
rrange = round_nearest(rrange,ROTATION_RANGE_PRECISION);
l_thetas = permute(l_thetas,[1,3,2]);
end

function xr = round_nearest(x,precision)

if isnan(x),
  xr = nan;
else
  xr = max(1,round(x/precision))*precision;
end

end

function [all_labels_horzflipped,all_labels_vertflipped] = flip_labels(all_labels,lobj)

  % all_labels is nkpts x d x nlabels
  templateidx1 = find(all(all(~isnan(all_labels),1),2),1);
  if isempty(templateidx1),
    templateidx1 = 1;
  end
  centroid = mean(all_labels,1,'omitmissing');
  all_labels_centered = all_labels - centroid;

  kptmatches = lobj.flipLandmarkMatches;

  ntries = 10;
  nlabels = size(all_labels,3);
  templateidx_rest = randsample(nlabels,min(ntries,nlabels),false);
  bestd_horz = inf;
  bestd_vert = inf;
  for templateidx = [templateidx1,templateidx_rest'],
    [all_labels_horzflipped_curr,d_horz] = flip_labels_helper(1,kptmatches,templateidx,centroid,all_labels_centered);
    [all_labels_vertflipped_curr,d_vert] = flip_labels_helper(2,kptmatches,templateidx,centroid,all_labels_centered);
    d_horz = median(d_horz);
    d_vert = median(d_vert);
    if d_horz < bestd_horz,
      all_labels_horzflipped = all_labels_horzflipped_curr;
    end
    if d_vert < bestd_vert,
      all_labels_vertflipped = all_labels_vertflipped_curr;
    end    
  end

end

function [all_labels_flipped,d_flipped] = flip_labels_helper(dim,kptmatches,templateidx,centroid,all_labels_centered)

  all_labels_flipped = all_labels_centered;
  all_labels_flipped(:,dim,:) = -all_labels_centered(:,dim,:);
  all_labels_flipped(kptmatches(:,1),dim,:) = -all_labels_centered(kptmatches(:,2),dim,:);
  all_labels_flipped(kptmatches(:,2),dim,:) = -all_labels_centered(kptmatches(:,1),dim,:);
  % normalize so that it is cos,sin 
  z = sqrt(sum(all_labels_flipped.^2,2));
  all_labels_flipped_normed = all_labels_flipped ./ z;
  z = sqrt(sum(all_labels_centered.^2,2));
  all_labels_centered_normed = all_labels_centered ./ z;

  d_flipped = sum(sum((all_labels_flipped_normed-all_labels_centered_normed(:,:,templateidx)).^2,2),1);
  d = sum(sum((all_labels_centered_normed-all_labels_centered_normed(:,:,templateidx)).^2,2),1);
  idx = d<=d_flipped;
  all_labels_flipped(:,:,idx) = all_labels_centered(:,:,idx);
  all_labels_flipped = all_labels_flipped + centroid;
  d_flipped(idx) = d(idx);
end
