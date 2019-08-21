function abdomen_vs_thorax_angle = compute_AbdomenAngle(pts,varargin)

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

% thorax to abdomen angle - negative is to the left, positive is to the
% right
abdomen_angle = atan2(pts(2,:,:,thorax_pt)-pts(2,:,:,abdomen_pt),pts(1,:,:,thorax_pt)-pts(1,:,:,abdomen_pt));
abdomen_vs_thorax_angle = modrange(thorax_angle-abdomen_angle,-pi,pi);

if DEBUG,
  
  nsample = 10;
  nlandmarks = size(pts,4);
  
  hfig = figure;
  clf;
  set(hfig,'Position',[10,10,1500,200]);
  nax = 1;
  hax = createsubplots(nax,nsample,0);
  hax = reshape(hax,[nax,nsample]);
  vwi = 1;

  bounds = prctile(abdomen_vs_thorax_angle,[1,99]);
  sampleangles = linspace(bounds(1),bounds(2),nsample);
  d = abs(modrange(abdomen_vs_thorax_angle'-sampleangles,-pi,pi));
  [mind,idxsample] = min(d,[],1);

  axi = 1;
  for exii = 1:nsample,
    exi = idxsample(exii);
    imagesc(gtimdata.ppdata.I{exi,vwi},'Parent',hax(axi,exii));
    axis(hax(axi,exii),'image','off');
    hold(hax(axi,exii),'on');
    for pti = 1:nlandmarks,
      plot(hax(axi,exii),pts(1,exi,1,pti),pts(2,exi,1,pti),'r.');
    end
    plot(hax(axi,exii),[meanshoulder(1,exi),pts(1,exi,1,thorax_pt)],[meanshoulder(2,exi),pts(2,exi,1,thorax_pt)],'c.-');
    plot(hax(axi,exii),squeeze(pts(1,exi,1,[thorax_pt,abdomen_pt])),squeeze(pts(2,exi,1,[thorax_pt,abdomen_pt])),'c.-');
    text(1,1,sprintf('%.1f deg',sampleangles(exii)*180/pi),...
      'HorizontalAlignment','left','VerticalAlignment','top','Parent',hax(axi,exii),'FontSize',6,'Color','c');
  end
  
  
  colormap gray;
  
end