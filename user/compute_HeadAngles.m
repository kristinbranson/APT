function [originpt,rot_theta,rot_pts,headang_midthorax2midhead,noseang_midthorax2nose] = compute_HeadAngles(pts)
% compute the angle from midpoint of shoulders to nose and midpoint of
% shoulder to midpoint of head (not used - too noisy)
% Alice Robie 4/16/2019
% pts input [xy, frms, flies, landmarks] so 2 x nfrms x nflies x 17 landmarks
% trx - ctrax output data
% code from acriptlist_APT_HeadAngle
% output ::
% originpt - thorax_mid
% rot_theta - angle of thorax mid to ref point (notum, 6)
% rot_pts nflies, d, 19 = 17 landmarks + 18 thoraxmid, 19 headmid, nframes
% headang_midthorax2midhead - 
% noseang_midthorax2nose



% if flies = 1 (like for GT table results)
if size(pts,3) == 1
% calculate mid thorax based on average of shoulders
thorax_mid(1,:,1) = squeeze((pts(1,:,:,4)+pts(1,:,:,5)))./2;
thorax_mid(2,:,1) = squeeze(pts(2,:,:,4)+pts(2,:,:,5))./2;
pts(:,:,1,18) = thorax_mid;    
% calculate mid head based on average of back of head 
head_mid(1,:,:) = squeeze((pts(1,:,:,2)+pts(1,:,:,3)))./2;
head_mid(2,:,:) = squeeze((pts(2,:,:,2)+pts(2,:,:,3)))./2;
pts(:,:,:,19) = head_mid;
else
    % calculate mid thorax based on average of shoulders
thorax_mid(1,:,:) = squeeze((pts(1,:,:,4)+pts(1,:,:,5)))./2;
thorax_mid(2,:,:) = squeeze(pts(2,:,:,4)+pts(2,:,:,5))./2;
pts(:,:,:,18) = thorax_mid;  
% calculate mid head based on average of back of head 
head_mid(1,:,:) = squeeze((pts(1,:,:,2)+pts(1,:,:,3)))./2;
head_mid(2,:,:) = squeeze((pts(2,:,:,2)+pts(2,:,:,3)))./2;
pts(:,:,:,19) = head_mid;
end

% set origin point
originpt = thorax_mid;
% set reference pt for center line
refpt = 6;

% substract off origin
ct_pts = pts-originpt;
% ct_thoraxmid = thorax_mid-originpt;
% ct_headmid = head_mid-originpt;
% figure,
% plot(squeeze(ct_pts(1,10,1,:)),squeeze(ct_pts(2,10,1,:)),'.'),axis ij


[d,frm,flies,ldmks] = size(pts);
% for each traj calculate angles
mx = [];
my = [];
theta = [];
pts_in = [];
rot_pts = [];
rot_theta = [];
headang_midthorax2midhead = [];
noseang_midthorax2nose = [];
for fly = 1:flies
    mx= squeeze(ct_pts(1,:,fly,refpt));
    my= squeeze(ct_pts(2,:,fly,refpt));
    theta = atan2(my,mx);
    
    % for a given fly: pts_in =  [x,y landmarks, frms]
    pts_in = permute(squeeze(ct_pts(:,:,fly,:)),[1,3,2]);
    
    % rot_pts = [fly, xy, landmakrs, frms]
    % align thorax midline to the negative y-axis (image ij)
    % [-theta rotates angle to % refpt (notum) to the pos x axis (nose
    % points left)
    % +pi/2 rotates nose to negative y-axis (up in image ij)]
    rot_pts(fly,1,:,:) = squeeze(pts_in(1,:,:)).*cos(-theta+pi/2) -squeeze(pts_in(2,:,:)).*sin(-theta+pi/2);
    rot_pts(fly,2,:,:) = squeeze(pts_in(1,:,:)).*sin(-theta+pi/2) +squeeze(pts_in(2,:,:)).*cos(-theta+pi/2);
    
    rot_theta(fly,1,:) = theta;
    
    %thorax midline to head midline angle
    % (head midline rot_pts ld #19)
    hx = squeeze(rot_pts(fly,1,19,:));
    hy = squeeze(rot_pts(fly,2,19,:));
    % add pi/2 to make centered at zero
    headang_midthorax2midhead(fly,:) = atan2(hy,hx)+pi/2;
    
    %thorax midline to nose (ld #1)
    nx = squeeze(rot_pts(fly,1,1,:));
    ny = squeeze(rot_pts(fly,2,1,:));
    % add pi/2 to make centered at zero
    noseang_midthorax2nose(fly,:) = atan2(ny,nx)+pi/2;
    
end