function [err_tables,ll] = label_outliers(lobj)

%%
ll = [];
ff = [];
mm = [];
tt = [];
for ndx = 1:numel(lobj.labels)
  ll = cat(2,ll, lobj.labels{ndx}.p);
  ff = cat(1, ff,lobj.labels{ndx}.frm);
  tt = cat(1,tt,lobj.labels{ndx}.tgt);
  mm(end+1:end+numel(lobj.labels{ndx}.frm)) = ndx;
end
all_ll = reshape(ll,size(ll,1)/2/lobj.nview,lobj.nview,2,[]);

npts = size(all_ll,1);
nlbl = size(all_ll,4);
edges = lobj.skeletonEdges;
if isempty(edges)
  G = graph(ones(npts));
  warning('No skeleton defined. Suspected label errors may flag a lot of correct labels');
  % if not skeleton then use all the triads
else
  G = graph(edges(:,1),edges(:,2));
end
nview = lobj.nview;
multi_view = nview>1;
%%

sus = false(1,nlbl);
sus_idx = zeros(1,nlbl);
sus_val = zeros(1,nlbl);
sus_view = zeros(1,nlbl);

for vw = 1:nview
ll = permute(all_ll(:,vw,:,:),[1,3,4,2]);
% compute all the angle for triads
all_ang = zeros(0,size(ll,3));
ix = zeros(0,3);
for a1 = 1:npts
  for b1 = 1:npts
    if a1==b1 continue; end
    [spath,plen] = shortestpath(G,a1,b1);
    if plen>2 continue; end
    % if a1 and b1 are far then ignore the triad.

    for c1 = a1:npts
      [spath,plen] = shortestpath(G,c1,b1);
      if plen>2 continue; end
      % if c1 and b1 are far then ignore the triad.
      if c1==b1 continue; end
      if c1==a1 continue; end
      p1 = squeeze(ll(a1,:,:));
      p2 = squeeze(ll(b1,:,:));
      p3 = squeeze(ll(c1,:,:));
      ang1 = atan2(p1(2,:)-p2(2,:),p1(1,:)-p2(1,:));
      ang2 = atan2(p3(2,:)-p2(2,:),p3(1,:)-p2(1,:));
      ang = mod(ang1-ang2,2*pi);
      all_ang(end+1,:) = ang;
      ix(end+1,:) = [a1,b1,c1];
    end
  end
end

%
out_i = zeros(0,2);

% suspiciousness threshold in degree
rth1 = 90;

% sus == suspiciousnes
for ndx = 1:size(all_ang,1)
  % since angles wrap around at 360, finding low threshold and high threshold are
  % not useful to find outliers. However, if the angle are shifted by 10,
  % 20.. and 350 degrees, and if we find the outliers for each of  these shifted
  % angles and the combine these outliers, then we can identify all the outliers
  
  cur_sus_val = zeros(1,nlbl);
  cur_sus = false(1,nlbl);
  cur_sus_idx = ones(1,nlbl);

  for offset = 0:10:350
    theta = all_ang(ndx,:)*180/pi;
    thetao = mod(theta + offset,360);
    th_max = prctile(thetao,98,2);
    th_min = prctile(thetao,2,2);
    rth = max(rth1,th_max-th_min);
    if (th_min-rth)<0 || (th_max+rth)>360
      continue
    end
    if any(thetao>(th_max+rth)) || any(thetao<(th_min-rth))      
      sus_high = (thetao>(th_max+rth));
      sus1_val = min(thetao-(th_max+rth),(360+th_min-rth)-thetao);
      sel = (sus1_val>sus_val)&sus_high;
      sus_idx(sel) = ndx;
      sus_val(sel) = sus1_val(sel);
      sus(sel) = true;

      sus_low = (thetao<(th_min-rth));
      sus1_val = min(360+thetao-(th_max+rth),thetao-(th_min-rth));
      sel = (sus1_val>sus_val)&sus_low;
      sus_idx(sel) = ndx;
      sus_val(sel) = sus1_val(sel);
      sus(sel) = true;
      sus_view(sel) = vw;
    end
  end
end
end
angle_table = table(mm(sus)',ff(sus),tt(sus),ix(sus_idx(sus),:),sus_val(sus)');
angle_table.Properties.VariableNames = {'Mov','Frm','Lbl', 'Triad','Susp'};
if multi_view
  angle.View = sus_view(sus)';
end

%% Distance outliers

pdist= zeros(size(ll,3),2,0);
px = zeros(0,3);

for vw = 1:nview
ll = permute(all_ll(:,vw,:,:),[1,3,4,2]);

xorr = zeros(0,2);
ixd = zeros(0,4);
for a1 = 1:npts
  for b1 = a1+1:npts
        [spath,plen] = shortestpath(G,a1,b1);
        if plen>2 continue; end

    for c1 = 1:npts
      for d1 = c1+1:npts
        if c1==a1 && d1==b1, continue; end
        [spath,plen] = shortestpath(G,c1,d1);
        if plen>2 continue; end
        p1 = squeeze(ll(a1,:,:));
        p2 = squeeze(ll(b1,:,:));
        p3 = squeeze(ll(c1,:,:));
        p4 = squeeze(ll(d1,:,:));
        dd1 = sqrt(sum( (p1-p2).^2,1));
        dd2 = sqrt(sum( (p3-p4).^2,1));
        zz = ~(isnan(dd1)|isnan(dd2));
        mean_X = nanmean(dd2);
        mean_Y = nanmean(dd1);

        % Calculate the estimated slope (beta1)
        numerator = sum((dd2(zz) - mean_X) .* (dd1(zz) - mean_Y));
        denominator = sum((dd2(zz) - mean_X).^2);
        beta1 = numerator / denominator;
        
        % Calculate the estimated intercept (beta0)
        beta0 = mean_Y - beta1 * mean_X;
        xorr(end+1,:) = [beta0,beta1];
        ixd(end+1,:) = [a1,b1,c1,d1];
      end
    end
  end
end

%%

for a1 = 1:npts
  for b1 = a1+1:npts
    [spath,plen] = shortestpath(G,a1,b1);
    if plen>2 continue; end
    p1 = squeeze(ll(a1,:,:));
    p2 = squeeze(ll(b1,:,:));
    dd1 = sqrt(sum( (p1-p2).^2,1));
    dd1m = dd1-nanmean(dd1);
    curp = zeros(0,size(ll,3));
    count = 0;
    for c1 = 1:npts
      for d1 = c1+1:npts
        if c1==a1 && d1==b1, continue; end
        [spath,plen] = shortestpath(G,c1,d1);
        if plen>2 continue; end
        p3 = squeeze(ll(c1,:,:));
        p4 = squeeze(ll(d1,:,:));
        dd2 = sqrt(sum( (p3-p4).^2,1));
        dd2m = dd2-nanmean(dd2);
        xndx = find(ismember(ixd,[a1,b1,c1,d1],'rows'));
        curp(end+1,:) = xorr(xndx,2)*dd2+xorr(xndx,1);
      end
    end
    curp = nanmean(curp,1);
    pdist(:,:,end+1) = [dd1; curp]';
    px(end+1,:) = [a1,b1,vw];
  end
end
end

xq = pdist(:,1,:)./pdist(:,2,:);
zq = xq>4;
xs = xq.*zq;
zs = max(xs,[],3);
ns = squeeze(sum(isnan(ll(:,1,:)),1));
sus = (zs>0)&(ns<npts/2);
sus_idx_vw = px(argmax(xs(sus,:,:),[],3),:);
sus_idx = sus_idx_vw(:,1:2);
sus_vw = sus_idx_vw(:,3);
dist_table = table(mm(sus)',ff(sus),tt(sus),sus_idx,zs(sus));
dist_table.Properties.VariableNames = {'Mov','Frm','Lbl', 'Segment','Susp'};
if multi_view
  dist_table.View = sus_vw;
end

err_tables = {angle_table,dist_table};



end