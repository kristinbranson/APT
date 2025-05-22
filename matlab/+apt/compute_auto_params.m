function autoparams = compute_auto_params(lobj)

  assert(lobj.nview==1, 'Auto setting of parameters not tested for multivew');
  %%
  view = 1;
  autoparams = containers.Map();
  nmov = numel(lobj.labels);
  npts = lobj.nLabelPoints;
  all_labels = [];
  all_id = [];
  all_mov = [];
  pair_labels = [];
  cur_pts = ((view-1)*2*npts+1):(view*2*npts);
  for ndx = 1:nmov        
    if ~Labels.hasLbls(lobj.labels{ndx}), continue; end

    all_labels = [all_labels lobj.labels{ndx}.p(cur_pts,:,:)];
    n_labels = size(lobj.labels{ndx}.p,2);
    all_id = [all_id 1:n_labels];
    all_mov = [all_mov ones(1,n_labels)*ndx];

    if ~lobj.maIsMA, continue, end
    pair_done = [];
    big_val = 10000000;
    for fndx = 1:numel(lobj.labels{ndx}.frm)
      f = lobj.labels{ndx}.frm(fndx);
      if nnz(lobj.labels{ndx}.frm==f)>1
        idx = find(lobj.labels{ndx}.frm==f);
        for ix = idx(:)'
          if ix==fndx, continue, end
          if any(pair_done==(f*big_val+ix))
            continue
          end
          if any(pair_done==(ix*big_val+f))
            continue
          end

          pair_done(end+1) = ix*big_val+f;
          cur_pair = [lobj.labels{ndx}.p(cur_pts,fndx);lobj.labels{ndx}.p(cur_pts,ix)];
          pair_labels = [pair_labels cur_pair];

        end
      end

    end

  end
  all_labels = reshape(all_labels,npts,2,[]);
  pair_labels = reshape(pair_labels,npts,2,2,[]);
  % animals are along the last dimension.
  % second dim has the coordinates

  %%

  l_min = permute(nanmin(all_labels,[],1),[2,3,1]);
  l_max = permute(nanmax(all_labels,[],1),[2,3,1]);
  l_span = l_max-l_min;

  multi_bbox_scale = isfield(lobj.trackParams.ROOT.MultiAnimal.TargetCrop,'multi_scale_by_bbox') && ...
    lobj.trackParams.ROOT.MultiAnimal.TargetCrop.multi_scale_by_bbox;
  l_span_pc = prctile(l_span,95,2);
  if ~multi_bbox_scale
    %l_min = reshape(min(all_labels,[],1),size(all_labels,[2,3]));
    %l_max = reshape(max(all_labels,[],1),size(all_labels,[2,3]));
    % l_span is labels span in x and y direction
  
    l_span_pc = prctile(l_span,95,2);
    l_span_max = nanmax(l_span,[],2);
  
    % Check and flag outliers..
    if any( (l_span_max./l_span_pc)>2)
      outliers = zeros(0,3);
      for jj = find( (l_span_max/l_span_pc)>2)
        ix = find(l_span(jj,:)>l_span_pc(jj)*2);
        for xx = ix(:)'
          mov = all_mov(xx);
          yy = all_id(xx);
          cur_fr = lobj.labels{mov}.frm(yy);
          cur_tgt = lobj.labels{mov}.tgt(yy);
          outliers(end+1,:) = [mov,cur_fr,cur_tgt];
        end
      end
      wstr = 'Some bounding boxes have sizes much larger than normal. This suggests that they may have labeing errors\n';
      wstr = sprintf('%s The list of examples is \n',wstr);
      for zz = 1:size(outliers,1)
        wstr = sprintf('%s Movie:%d, frame:%d, target:%d\n',wstr,outliers(zz,1),outliers(zz,2),outliers(zz,3));
      end
      warning(wstr);
    end
  
    crop_radius = nanmax(l_span_pc);
    crop_radius = ceil(crop_radius/16)*16;
  else
    l_span_median = prctile(l_span,50,2);
    crop_radius = nanmax(l_span_median);
    crop_radius = ceil(crop_radius/16)*16;   
  end
  autoparams('MultiAnimal.TargetCrop.ManualRadius') = crop_radius;
  if ~lobj.trackerIsTwoStage && ~lobj.hasTrx
    autoparams('MultiAnimal.TargetCrop.AlignUsingTrxTheta') = false;
  end

  % Look at distances between labeled pairs to find what to set for
  % tranlation range for data augmentation
  min_pairs = min(pair_labels,[],1);
  max_pairs = max(pair_labels,[],1);
  ctr_pairs = (max_pairs+min_pairs)/2;
  d_pairs = sqrt(sum( (ctr_pairs(:,:,1,:)-ctr_pairs(:,:,2,:)).^2,2));
  d_pairs = squeeze(d_pairs);
  d_pairs_pc = prctile(d_pairs,5);
  d_pairs_min = min(d_pairs_pc);

  % If the distances between center of animals is much less than the
  % span then warn about using bbox based methods
  if(d_pairs_pc<min(l_span_pc/10)) && lobj.trackerIsObjDet
    wstr = 'The distances between the center of animals is much smaller than the size of the animals';
    wstr = sprintf('%s\n Avoid using object detection based top-down methods',wstr);
    warning(wstr);  %#ok<SPWRN>
  end

  trange_frame = min(lobj.movienr(view),lobj.movienc(view))/10;
  trange_pair = d_pairs_pc/2;
  trange_crop = min(l_span_pc)/10; 
  trange_animal = min(trange_pair,trange_crop);
  rrange = 15;
  align_theta = lobj.trackParams.ROOT.MultiAnimal.TargetCrop.AlignUsingTrxTheta;

  % Estimate the translation range and rotation ranges
  if ~lobj.maIsMA
    % Single animal autoparameters.
    if lobj.hasTrx
      % If has trx then just use defaults for rrange
      % If using cropping set trange to 10 percent of the crop size.
      trange = max(5,trange_animal);
      trange = round(trange/5)*5;
      if lobj.trackParams.ROOT.MultiAnimal.TargetCrop.AlignUsingTrxTheta
        rrange = 15;
      else
        rrange = 180;
      end
      autoparams('DeepTrack.DataAugmentation.rrange') = rrange;
      autoparams('DeepTrack.DataAugmentation.trange') = trange;
    else
      % No trx. 

      % Set translation range to 10 percent of the crop size
      trange = max(5,trange_frame);
      trange = round(trange/5)*5;
      % Try to guess rotation range for single animal
      % For this look at the angles from center of the frame/crop to
      % the labels
      l_thetas = atan2(all_labels(:,1,:)-lobj.movienc(view)/2,...
        all_labels(:,2,:)-lobj.movienr(view)/2);
      ang_span = get_angle_span(l_thetas)*180/pi;
      rrange = min(180,max(10,median(ang_span)/2));
      rrange = round(rrange/10)*10;
      autoparams('DeepTrack.DataAugmentation.rrange') = rrange;
      autoparams('DeepTrack.DataAugmentation.trange') = trange;
    end
  else
    % Multi-animal. Two tranges for first and second stage. Applied
    % depending on the workflow

    
    if lobj.trackParams.ROOT.MultiAnimal.multi_crop_ims
      % If we are cropping the images then use animal trange.
      crop_sz = lobj.trackParams.ROOT.MultiAnimal.multi_crop_im_sz;
      trange_crop = crop_sz/10;
      trange_top = min(trange_pair,trange_crop);
    else
      trange_top = trange_frame;
    end
    
    trange_top = max(5,trange_top);
    trange_top = round(trange_top/5)*5;
    trange_second = min(trange_crop,trange_animal);
    trange_second = max(5,trange_second);
    trange_second = round(trange_second/5)*5;

    if lobj.trackerIsTwoStage
      % For 2 stage DeepTrack.DataAugmentation is for the second stage
      % and Multianiml.Detect.DeepTrack.DataAugmentation is for the first
      % stage. Note DataAugmentation doesn't apply for bbox detection
      autoparams('DeepTrack.DataAugmentation.trange') = trange_second;
      if lobj.trackerIsObjDet
        % If uses object detection for the first stage then the look at
        % the variation in angles relative to the center
        mid_labels = mean(all_labels,1);
        l_thetas = atan2(all_labels(:,1,:)-mid_labels(:,1,:),...
          all_labels(:,2,:)-mid_labels(:,2,:));
        ang_span = get_angle_span(l_thetas)*180/pi;
        rrange = min(180,max(10,median(ang_span)/2));
        rrange = round(rrange/10)*10;
        autoparams('DeepTrack.DataAugmentation.rrange') = rrange;
      else
        % Using head-tail for the first stage
        align_trx_theta = lobj.trackParams.ROOT.MultiAnimal.TargetCrop.AlignUsingTrxTheta;

        if align_trx_theta
        % For head-tail based, find the angle span after aligning along
        % the head-tail direction.

          hd = all_labels(lobj.skelHead,:,:);
          tl = all_labels(lobj.skelTail,:,:);
          body_ctr = (hd+tl)/2;
          l_thetas = atan2(all_labels(:,1,:)-body_ctr(:,1,:),...
            all_labels(:,2,:)-body_ctr(:,2,:));
          l_thetas_r = mod(l_thetas - l_thetas(lobj.skelHead,:,:),2*pi);
          ang_span = get_angle_span(l_thetas_r)*180/pi;
          
          % Remove ang spans for head and tail because they will be always
          % zero.
          ht_pts = [lobj.skelHead, lobj.skelTail];
          ang_span(ht_pts) = [];
        else
          % If not aligned along the head-tail then look at the angles
          % from the center
          warning('For head-tail based two-stage detection, align using head-tail is switched off. Aligning using head-tail will lead to better performance');
          mid_labels = mean(all_labels,1);
          l_thetas = atan2(all_labels(:,1,:)-mid_labels(:,1,:),...
            all_labels(:,2,:)-mid_labels(:,2,:));
          ang_span = get_angle_span(l_thetas)*180/pi;
        end
        if numel(ang_span)>0
          rrange = min(180,max(10,median(ang_span)/2));
        else
          rrange = 15;
        end
        rrange = round(rrange/10)*10;
        autoparams('DeepTrack.DataAugmentation.rrange') = rrange;
        
        mid_labels = mean(all_labels,1);
        l_thetas = atan2(all_labels(:,1,:)-mid_labels(:,1,:),...
          all_labels(:,2,:)-mid_labels(:,2,:));
        ang_span = get_angle_span(l_thetas)*180/pi;
        ang_span = ang_span([lobj.skelHead,lobj.skelTail]);
        rrange = min(180,max(10,median(ang_span)/2));
        rrange = round(rrange/10)*10;
        autoparams('MultiAnimal.Detect.DeepTrack.DataAugmentation.rrange') = rrange;
        autoparams('MultiAnimal.Detect.DeepTrack.DataAugmentation.trange') = trange_top;
      end
    else
      % Bottom up. Just look at angles of landmarks to the center to
      % see if the animals tend to be always aligned.
      mid_labels = mean(all_labels,1);
      l_thetas = atan2(all_labels(:,1,:)-mid_labels(:,1,:),...
        all_labels(:,2,:)-mid_labels(:,2,:));
      ang_span = get_angle_span(l_thetas)*180/pi;
      rrange = min(180,max(10,median(ang_span)/2));
      rrange = round(rrange/10)*10;
      autoparams('DeepTrack.DataAugmentation.rrange') = rrange;
      autoparams('DeepTrack.DataAugmentation.trange') = trange_top;

    end  % if
  end  % if 

end  % function



function ang_span = get_angle_span(theta)
  % Find the span of thetas. Hacky method that rotates the pts by
  % 10degrees and then checks the span.
  ang_span = ones(size(theta,1),1)*2*pi;
  for offset = 0:10:360
    thetao = mod(theta + offset*pi/180,2*pi);
    cur_span = prctile(thetao,98,3) - prctile(thetao,2,3);
    ang_span = min(ang_span,cur_span);
  end
end

