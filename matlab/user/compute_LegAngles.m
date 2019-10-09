function out = compute_LegAngles(pts,varargin)

[DEBUG] = myparse(varargin,'debug',false);

% pts input [xy, frms, flies, landmarks] so 2 x nfrms x nflies x 17 landmarks
left_shoulder_pt = 5;
right_shoulder_pt = 4;
shoulder_pts = [left_shoulder_pt,right_shoulder_pt];
thorax_pt = 6;
right_middle_femur_pt = 10;
right_middle_tibia_pt = 11;
right_middle_tarsus_pt = 16;
left_middle_femur_pt = 8;
left_middle_tibia_pt = 9;
left_middle_tarsus_pt = 13;
left_front_tarsus_pt = 12;
left_rear_tarsus_pt = 14;
right_front_tarsus_pt = 17;
right_rear_tarsus_pt = 15;
abdomen_pt = 7;

% compute thorax angle
meanshoulder = mean(pts(:,:,:,shoulder_pts),4);
thorax_angle = atan2(meanshoulder(2,:,:,:)-pts(2,:,:,thorax_pt),meanshoulder(1,:,:,:)-pts(1,:,:,thorax_pt));

% compute left middle femur angle
% add pi/2 so that it is relative to angle orthogonal to thorax
% angle
% negative values mean the tibia is toward the head, positive values mean
% the tibia is toward the tail
leftmiddlefemur_angle = atan2(pts(2,:,:,left_middle_tibia_pt)-pts(2,:,:,left_middle_femur_pt),pts(1,:,:,left_middle_tibia_pt)-pts(1,:,:,left_middle_femur_pt));
leftmiddlefemur_vs_thorax_angle = modrange(leftmiddlefemur_angle - thorax_angle + pi/2,-pi,pi);

% compute right middle femur angle
% subtract pi/2 so that it is relative to angle orthogonal to thorax
% angle
% positive values mean the tibia is toward the head, negative values mean
% the tibia is toward the tail
rightmiddlefemur_angle = atan2(pts(2,:,:,right_middle_tibia_pt)-pts(2,:,:,right_middle_femur_pt),pts(1,:,:,right_middle_tibia_pt)-pts(1,:,:,right_middle_femur_pt));
rightmiddlefemur_vs_thorax_angle = modrange(rightmiddlefemur_angle - thorax_angle - pi/2,-pi,pi);

% compute tarsus angles
leftfronttarsus_angle = atan2(pts(2,:,:,left_front_tarsus_pt)-pts(2,:,:,left_shoulder_pt),pts(1,:,:,left_front_tarsus_pt)-pts(1,:,:,left_shoulder_pt));
leftfronttarsus_vs_thorax_angle = modrange(leftfronttarsus_angle - thorax_angle + pi/2,-pi,pi);
rightfronttarsus_angle = atan2(pts(2,:,:,right_front_tarsus_pt)-pts(2,:,:,right_shoulder_pt),pts(1,:,:,right_front_tarsus_pt)-pts(1,:,:,right_shoulder_pt));
rightfronttarsus_vs_thorax_angle = modrange(rightfronttarsus_angle - thorax_angle - pi/2,-pi,pi);

leftmiddletarsus_angle = atan2(pts(2,:,:,left_middle_tarsus_pt)-pts(2,:,:,left_middle_femur_pt),pts(1,:,:,left_middle_tarsus_pt)-pts(1,:,:,left_middle_femur_pt));
leftmiddletarsus_vs_thorax_angle = modrange(leftmiddletarsus_angle - thorax_angle + pi/2,-pi,pi);
rightmiddletarsus_angle = atan2(pts(2,:,:,right_middle_tarsus_pt)-pts(2,:,:,right_middle_femur_pt),pts(1,:,:,right_middle_tarsus_pt)-pts(1,:,:,right_middle_femur_pt));
rightmiddletarsus_vs_thorax_angle = modrange(rightmiddletarsus_angle - thorax_angle - pi/2,-pi,pi);

leftreartarsus_angle = atan2(pts(2,:,:,left_rear_tarsus_pt)-pts(2,:,:,thorax_pt),pts(1,:,:,left_rear_tarsus_pt)-pts(1,:,:,thorax_pt));
leftreartarsus_vs_thorax_angle = modrange(leftreartarsus_angle - thorax_angle + pi/2,-pi,pi);
rightreartarsus_angle = atan2(pts(2,:,:,right_rear_tarsus_pt)-pts(2,:,:,thorax_pt),pts(1,:,:,right_rear_tarsus_pt)-pts(1,:,:,thorax_pt));
rightreartarsus_vs_thorax_angle = modrange(rightreartarsus_angle - thorax_angle - pi/2,-pi,pi);

% compute left middle leg bend
% negative values are toward the front, positive values are toward the back
leftmiddlefemur_angle = atan2(pts(2,:,:,left_middle_tibia_pt)-pts(2,:,:,left_middle_femur_pt),pts(1,:,:,left_middle_tibia_pt)-pts(1,:,:,left_middle_femur_pt));
lefttibia_angle = atan2(pts(2,:,:,left_middle_tarsus_pt)-pts(2,:,:,left_middle_tibia_pt),pts(1,:,:,left_middle_tarsus_pt)-pts(1,:,:,left_middle_tibia_pt));
leftmiddleknee_angle = modrange(leftmiddlefemur_angle - lefttibia_angle,-pi,pi);

% compute right middle leg bend
% negative values are toward the front, positive values are toward the back
rightmiddlefemur_angle = atan2(pts(2,:,:,right_middle_tibia_pt)-pts(2,:,:,right_middle_femur_pt),pts(1,:,:,right_middle_tibia_pt)-pts(1,:,:,right_middle_femur_pt));
righttibia_angle = atan2(pts(2,:,:,right_middle_tarsus_pt)-pts(2,:,:,right_middle_tibia_pt),pts(1,:,:,right_middle_tarsus_pt)-pts(1,:,:,right_middle_tibia_pt));
rightmiddleknee_angle = modrange(rightmiddlefemur_angle - righttibia_angle,-pi,pi);

out = struct;
out.leftfronttarsus_vs_thorax_angle = leftfronttarsus_vs_thorax_angle;
out.rightfronttarsus_vs_thorax_angle = rightfronttarsus_vs_thorax_angle;
out.leftmiddletarsus_vs_thorax_angle = leftmiddletarsus_vs_thorax_angle;
out.rightmiddletarsus_vs_thorax_angle = rightmiddletarsus_vs_thorax_angle;
out.leftreartarsus_vs_thorax_angle = leftreartarsus_vs_thorax_angle;
out.rightreartarsus_vs_thorax_angle = rightreartarsus_vs_thorax_angle;
out.leftmiddleknee_angle = leftmiddleknee_angle;
out.rightmiddleknee_angle = rightmiddleknee_angle;
out.leftmiddlefemur_vs_thorax_angle = leftmiddlefemur_vs_thorax_angle;
out.rightmiddlefemur_vs_thorax_angle = rightmiddlefemur_vs_thorax_angle;


if DEBUG,
  
  nsample = 10;
  nlandmarks = size(pts,4);
  
  hfig = figure;
  clf;
  set(hfig,'Position',[10,10,1500,1400]);
  nax = 10;
  hax = createsubplots(nax,nsample,0);
  hax = reshape(hax,[nax,nsample]);
  vwi = 1;

  bounds = prctile(cat(2,-leftfronttarsus_vs_thorax_angle,rightfronttarsus_vs_thorax_angle),[1,99]);
  sampleangles = linspace(bounds(1),bounds(2),nsample);
  d = abs(modrange(-leftfronttarsus_vs_thorax_angle'-sampleangles,-pi,pi));
  [mind,leftidxsample] = min(d,[],1);
  d = abs(modrange(rightfronttarsus_vs_thorax_angle'-sampleangles,-pi,pi));
  [mind,rightidxsample] = min(d,[],1);

  axi = 1;
  for exii = 1:nsample,
    exi = leftidxsample(exii);
    imagesc(gtimdata.ppdata.I{exi,vwi},'Parent',hax(axi,exii));
    axis(hax(axi,exii),'image','off');
    hold(hax(axi,exii),'on');
    for pti = 1:nlandmarks,
      plot(hax(axi,exii),pts(1,exi,1,pti),pts(2,exi,1,pti),'r.');
    end
    plot(hax(axi,exii),squeeze(pts(1,exi,1,[left_front_tarsus_pt,left_shoulder_pt])),squeeze(pts(2,exi,1,[left_front_tarsus_pt,left_shoulder_pt])),'c.-');
    plot(hax(axi,exii),squeeze(pts(1,exi,1,[right_front_tarsus_pt,right_shoulder_pt])),squeeze(pts(2,exi,1,[right_front_tarsus_pt,right_shoulder_pt])),'m.-');
    text(1,1,sprintf('%.1f deg',-sampleangles(exii)*180/pi),...
      'HorizontalAlignment','left','VerticalAlignment','top','Parent',hax(axi,exii),'FontSize',6,'Color','c');
    %set(hax(axi,exii),'YDir','normal');
  end
  
  axi = 2;
  for exii = 1:nsample,
    exi = rightidxsample(exii);
    imagesc(gtimdata.ppdata.I{exi,vwi},'Parent',hax(axi,exii));
    axis(hax(axi,exii),'image','off');
    hold(hax(axi,exii),'on');
    for pti = 1:nlandmarks,
      plot(hax(axi,exii),pts(1,exi,1,pti),pts(2,exi,1,pti),'r.');
    end
    plot(hax(axi,exii),squeeze(pts(1,exi,1,[left_front_tarsus_pt,left_shoulder_pt])),squeeze(pts(2,exi,1,[left_front_tarsus_pt,left_shoulder_pt])),'c.-');
    plot(hax(axi,exii),squeeze(pts(1,exi,1,[right_front_tarsus_pt,right_shoulder_pt])),squeeze(pts(2,exi,1,[right_front_tarsus_pt,right_shoulder_pt])),'m.-');
    text(1,1,sprintf('%.1f deg',sampleangles(exii)*180/pi),...
      'HorizontalAlignment','left','VerticalAlignment','top','Parent',hax(axi,exii),'FontSize',6,'Color','m');
    %set(hax(axi,exii),'YDir','normal');
  end
  
  bounds = prctile(cat(2,-leftmiddletarsus_vs_thorax_angle,rightmiddletarsus_vs_thorax_angle),[1,99]);
  sampleangles = linspace(bounds(1),bounds(2),nsample);
  d = abs(modrange(-leftmiddletarsus_vs_thorax_angle'-sampleangles,-pi,pi));
  [mind,leftidxsample] = min(d,[],1);
  d = abs(modrange(rightmiddletarsus_vs_thorax_angle'-sampleangles,-pi,pi));
  [mind,rightidxsample] = min(d,[],1);

  axi = 3;
  for exii = 1:nsample,
    exi = leftidxsample(exii);
    imagesc(gtimdata.ppdata.I{exi,vwi},'Parent',hax(axi,exii));
    axis(hax(axi,exii),'image','off');
    hold(hax(axi,exii),'on');
    for pti = 1:nlandmarks,
      plot(hax(axi,exii),pts(1,exi,1,pti),pts(2,exi,1,pti),'r.');
    end
    plot(hax(axi,exii),squeeze(pts(1,exi,1,[left_middle_tarsus_pt,left_middle_femur_pt])),squeeze(pts(2,exi,1,[left_middle_tarsus_pt,left_middle_femur_pt])),'c.-');
    plot(hax(axi,exii),squeeze(pts(1,exi,1,[right_middle_tarsus_pt,right_middle_femur_pt])),squeeze(pts(2,exi,1,[right_middle_tarsus_pt,right_middle_femur_pt])),'m.-');
    text(1,1,sprintf('%.1f deg',-sampleangles(exii)*180/pi),...
      'HorizontalAlignment','left','VerticalAlignment','top','Parent',hax(axi,exii),'FontSize',6,'Color','c');
    %set(hax(axi,exii),'YDir','normal');
  end
  
  axi = 4;
  for exii = 1:nsample,
    exi = rightidxsample(exii);
    imagesc(gtimdata.ppdata.I{exi,vwi},'Parent',hax(axi,exii));
    axis(hax(axi,exii),'image','off');
    hold(hax(axi,exii),'on');
    for pti = 1:nlandmarks,
      plot(hax(axi,exii),pts(1,exi,1,pti),pts(2,exi,1,pti),'r.');
    end
    plot(hax(axi,exii),squeeze(pts(1,exi,1,[left_middle_tarsus_pt,left_middle_femur_pt])),squeeze(pts(2,exi,1,[left_middle_tarsus_pt,left_middle_femur_pt])),'c.-');
    plot(hax(axi,exii),squeeze(pts(1,exi,1,[right_middle_tarsus_pt,right_middle_femur_pt])),squeeze(pts(2,exi,1,[right_middle_tarsus_pt,right_middle_femur_pt])),'m.-');
    text(1,1,sprintf('%.1f deg',sampleangles(exii)*180/pi),...
      'HorizontalAlignment','left','VerticalAlignment','top','Parent',hax(axi,exii),'FontSize',6,'Color','m');
    %set(hax(axi,exii),'YDir','normal');
  end

  bounds = prctile(cat(2,-leftreartarsus_vs_thorax_angle,rightreartarsus_vs_thorax_angle),[1,99]);
  sampleangles = linspace(bounds(1),bounds(2),nsample);
  d = abs(modrange(-leftreartarsus_vs_thorax_angle'-sampleangles,-pi,pi));
  [mind,leftidxsample] = min(d,[],1);
  d = abs(modrange(rightreartarsus_vs_thorax_angle'-sampleangles,-pi,pi));
  [mind,rightidxsample] = min(d,[],1);

  axi = 5;
  for exii = 1:nsample,
    exi = leftidxsample(exii);
    imagesc(gtimdata.ppdata.I{exi,vwi},'Parent',hax(axi,exii));
    axis(hax(axi,exii),'image','off');
    hold(hax(axi,exii),'on');
    for pti = 1:nlandmarks,
      plot(hax(axi,exii),pts(1,exi,1,pti),pts(2,exi,1,pti),'r.');
    end
    plot(hax(axi,exii),squeeze(pts(1,exi,1,[left_rear_tarsus_pt,thorax_pt])),squeeze(pts(2,exi,1,[left_rear_tarsus_pt,thorax_pt])),'c.-');
    plot(hax(axi,exii),squeeze(pts(1,exi,1,[right_rear_tarsus_pt,thorax_pt])),squeeze(pts(2,exi,1,[right_rear_tarsus_pt,thorax_pt])),'m.-');
    text(1,1,sprintf('%.1f deg',-sampleangles(exii)*180/pi),...
      'HorizontalAlignment','left','VerticalAlignment','top','Parent',hax(axi,exii),'FontSize',6,'Color','c');
    %set(hax(axi,exii),'YDir','normal');
  end
  
  axi = 6;
  for exii = 1:nsample,
    exi = rightidxsample(exii);
    imagesc(gtimdata.ppdata.I{exi,vwi},'Parent',hax(axi,exii));
    axis(hax(axi,exii),'image','off');
    hold(hax(axi,exii),'on');
    for pti = 1:nlandmarks,
      plot(hax(axi,exii),pts(1,exi,1,pti),pts(2,exi,1,pti),'r.');
    end
    plot(hax(axi,exii),squeeze(pts(1,exi,1,[left_rear_tarsus_pt,thorax_pt])),squeeze(pts(2,exi,1,[left_rear_tarsus_pt,thorax_pt])),'c.-');
    plot(hax(axi,exii),squeeze(pts(1,exi,1,[right_rear_tarsus_pt,thorax_pt])),squeeze(pts(2,exi,1,[right_rear_tarsus_pt,thorax_pt])),'m.-');
    text(1,1,sprintf('%.1f deg',sampleangles(exii)*180/pi),...
      'HorizontalAlignment','left','VerticalAlignment','top','Parent',hax(axi,exii),'FontSize',6,'Color','m');
    %set(hax(axi,exii),'YDir','normal');
  end
  
    bounds = prctile(cat(2,-leftmiddlefemur_vs_thorax_angle,rightmiddlefemur_vs_thorax_angle),[1,99]);
  sampleangles = linspace(bounds(1),bounds(2),nsample);
  d = abs(modrange(-leftmiddlefemur_vs_thorax_angle'-sampleangles,-pi,pi));
  [mind,leftidxsample] = min(d,[],1);
  d = abs(modrange(rightmiddlefemur_vs_thorax_angle'-sampleangles,-pi,pi));
  [mind,rightidxsample] = min(d,[],1);

  axi = 7;
  for exii = 1:nsample,
    exi = leftidxsample(exii);
    imagesc(gtimdata.ppdata.I{exi,vwi},'Parent',hax(axi,exii));
    axis(hax(axi,exii),'image','off');
    hold(hax(axi,exii),'on');
    for pti = 1:nlandmarks,
      plot(hax(axi,exii),pts(1,exi,1,pti),pts(2,exi,1,pti),'r.');
    end
    plot(hax(axi,exii),squeeze(pts(1,exi,1,[left_middle_femur_pt,left_middle_tibia_pt])),squeeze(pts(2,exi,1,[left_middle_femur_pt,left_middle_tibia_pt])),'c.-');
    plot(hax(axi,exii),squeeze(pts(1,exi,1,[right_middle_femur_pt,right_middle_tibia_pt])),squeeze(pts(2,exi,1,[right_middle_femur_pt,right_middle_tibia_pt])),'m.-');
    text(1,1,sprintf('%.1f deg',-sampleangles(exii)*180/pi),...
      'HorizontalAlignment','left','VerticalAlignment','top','Parent',hax(axi,exii),'FontSize',6,'Color','c');
    %set(hax(axi,exii),'YDir','normal');
  end

  axi = 8;
  for exii = 1:nsample,
    exi = rightidxsample(exii);
    imagesc(gtimdata.ppdata.I{exi,vwi},'Parent',hax(axi,exii));
    axis(hax(axi,exii),'image','off');
    hold(hax(axi,exii),'on');
    for pti = 1:nlandmarks,
      plot(hax(axi,exii),pts(1,exi,1,pti),pts(2,exi,1,pti),'r.');
    end
    plot(hax(axi,exii),squeeze(pts(1,exi,1,[left_middle_femur_pt,left_middle_tibia_pt])),squeeze(pts(2,exi,1,[left_middle_femur_pt,left_middle_tibia_pt])),'c.-');
    plot(hax(axi,exii),squeeze(pts(1,exi,1,[right_middle_femur_pt,right_middle_tibia_pt])),squeeze(pts(2,exi,1,[right_middle_femur_pt,right_middle_tibia_pt])),'m.-');
    text(1,1,sprintf('%.1f deg',sampleangles(exii)*180/pi),...
      'HorizontalAlignment','left','VerticalAlignment','top','Parent',hax(axi,exii),'FontSize',6,'Color','m');
    %set(hax(axi,exii),'YDir','normal');
  end

  
  
  bounds = prctile(cat(2,-leftmiddleknee_angle,rightmiddleknee_angle),[1,99]);
  sampleangles = linspace(bounds(1),bounds(2),nsample);
  d = abs(modrange(-leftmiddleknee_angle'-sampleangles,-pi,pi));
  [mind,leftidxsample] = min(d,[],1);
  d = abs(modrange(rightmiddleknee_angle'-sampleangles,-pi,pi));
  [mind,rightidxsample] = min(d,[],1);

  axi = 9;
  for exii = 1:nsample,
    exi = leftidxsample(exii);
    imagesc(gtimdata.ppdata.I{exi,vwi},'Parent',hax(axi,exii));
    axis(hax(axi,exii),'image','off');
    hold(hax(axi,exii),'on');
    for pti = 1:nlandmarks,
      plot(hax(axi,exii),pts(1,exi,1,pti),pts(2,exi,1,pti),'r.');
    end
    plot(hax(axi,exii),squeeze(pts(1,exi,1,[left_middle_tarsus_pt,left_middle_tibia_pt,left_middle_femur_pt])),squeeze(pts(2,exi,1,[left_middle_tarsus_pt,left_middle_tibia_pt,left_middle_femur_pt])),'c.-');
    plot(hax(axi,exii),squeeze(pts(1,exi,1,[right_middle_tarsus_pt,right_middle_tibia_pt,right_middle_femur_pt])),squeeze(pts(2,exi,1,[right_middle_tarsus_pt,right_middle_tibia_pt,right_middle_femur_pt])),'m.-');
    text(1,1,sprintf('%.1f deg',-sampleangles(exii)*180/pi),...
      'HorizontalAlignment','left','VerticalAlignment','top','Parent',hax(axi,exii),'FontSize',6,'Color','c');
    %set(hax(axi,exii),'YDir','normal');
  end
  
  axi = 10;
  for exii = 1:nsample,
    exi = rightidxsample(exii);
    imagesc(gtimdata.ppdata.I{exi,vwi},'Parent',hax(axi,exii));
    axis(hax(axi,exii),'image','off');
    hold(hax(axi,exii),'on');
    for pti = 1:nlandmarks,
      plot(hax(axi,exii),pts(1,exi,1,pti),pts(2,exi,1,pti),'r.');
    end
    plot(hax(axi,exii),squeeze(pts(1,exi,1,[left_middle_tarsus_pt,left_middle_tibia_pt,left_middle_femur_pt])),squeeze(pts(2,exi,1,[left_middle_tarsus_pt,left_middle_tibia_pt,left_middle_femur_pt])),'c.-');
    plot(hax(axi,exii),squeeze(pts(1,exi,1,[right_middle_tarsus_pt,right_middle_tibia_pt,right_middle_femur_pt])),squeeze(pts(2,exi,1,[right_middle_tarsus_pt,right_middle_tibia_pt,right_middle_femur_pt])),'m.-');
    text(1,1,sprintf('%.1f deg',sampleangles(exii)*180/pi),...
      'HorizontalAlignment','left','VerticalAlignment','top','Parent',hax(axi,exii),'FontSize',6,'Color','m');
    %set(hax(axi,exii),'YDir','normal');
  end
  
  
  colormap gray;
  
end