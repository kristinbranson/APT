function [q,trans,scaleFac,residuals,rotMat,axisAngleRad,EulerRPY]=computeRotations(refPoint,dataPoint)
%estimates rotation & translation between two clouds of points.  Returns
%rotation information in a varienty of formats.
%
% refPoint/dataPoint format = nPts x 3.  columns = xyz, rows = points.
%
% q = quaternion describing rotation.  Stored in format such that the axis
% of rotation is given by xyz = (q(2:4)./norm(q(2:4)))' and the angle rotated
% around that axis is given by 2*acos(q(1))  
% 
% trans = translation left over when rotation is done.
%
% scaleFac = scale factor required to make points match once rotation and
% translation are accounted for.
%
% residuals = distances between points once rotation, tranlation and
% scaling have been applied.
%
% rotMat = rotation in format of rotation matrix. Rotated points =
% (rotMat*dataPoint')'
%
% axisAngleRad = rotation in axis-angle format.  [x,y,z,angle] format.
% Angle is in radians.
%
% EulerRPY = Euler angles describing rotation.  XYZ axis order convention
% for Euler angles i.e. bank/roll->elevation/pitch->heading/yaw order.
% Radians
%
% You can transform ref point to try and match data point using:
%  transformedRefPoints = (scaleFac*rotMat*refPoint' + trans)';
% 
% Stephen's note: These parameters are calculated by subtracting the
% centroid from points in Ai, then rotating them using R around their
% central point/local origin, then translating them using T to the
% correct location.  This means you can't use R on its own without T to move points
% unless you either (1) first subtract the centroid of points to generate a local origin
% or (2) Your origin happens to correspond to the actual pivot point the
% points were rotated about in which case the axis and angle used to
% rotate points about their centroid-subtracted local coord system and
% the true pivot point are equivalent (I think).
%
% Uses Horn's method to estimate quaternion then converts to other rotation conventions:
%http://people.csail.mit.edu/bkph/papers/Absolute_Orientation.pdf
% @ARTICLE{Horn87closed-formsolution,
%     author = {Berthold K. P. Horn},
%     title = {Closed-form solution of absolute orientation using unit quaternions},
%     journal = {Journal of the Optical Society of America A},
%     year = {1987},
%     volume = {4},
%     number = {4},
%     pages = {629--642}
% }
%
% Dependencies: 
%       
%           https://www.mathworks.com/matlabcentral/fileexchange/35475-quaternions?s_tid=FX_rc2_behav
%           https://github.com/mattools/matGeom.git


%only set to 1 if debugging, 0 otherwise
debugPlots=0;

%% checking inputs

nPts = size(refPoint,1);

if nPts<4
    error('Need at least 4 points')
end

%% subtracting centroids

%Centroids
centRef = mean(refPoint,1);
centData = mean(dataPoint,1);

%subtract centroid
refZeroed = refPoint - repmat(centRef,nPts,1);
dataZeroed = dataPoint - repmat(centData,nPts,1);

if debugPlots ==1
    figure
    set(gcf,'visible','on')
    plot3(dataPoint(:,1),dataPoint(:,2),dataPoint(:,3),'bx')
    hold on
    plot3(centData(1),centData(2),centData(3),'bo')
    plot3(dataZeroed(:,1),dataZeroed(:,2),dataZeroed(:,3),'rx')
    shouldBeZero = mean(dataZeroed,1)
    plot3(shouldBeZero(1),shouldBeZero(2),shouldBeZero(3),'ro')
    plot3(0,0,0,'kx')
    axis equal
end

%% making matrix 'N' from Horn paper (p635) - matrix of sum of products of ref points and data points.  Used to compute rotations

%sums of products
Sxx = sum( refZeroed(:,1) .* dataZeroed(:,1) );
Sxy = sum( refZeroed(:,1) .* dataZeroed(:,2) );
Sxz = sum( refZeroed(:,1) .* dataZeroed(:,3) );

Syy = sum( refZeroed(:,2) .* dataZeroed(:,2) );
Syx = sum( refZeroed(:,2) .* dataZeroed(:,1) );
Syz = sum( refZeroed(:,2) .* dataZeroed(:,3) );

Szz = sum( refZeroed(:,3) .* dataZeroed(:,3) );
Szx = sum( refZeroed(:,3) .* dataZeroed(:,1) );
Szy = sum( refZeroed(:,3) .* dataZeroed(:,2) );


N = [...
    (Sxx+Syy+Szz),      Syz-Szy,        Szx-Sxz,        Sxy-Syx;...
    Syz-Szy,            (Sxx-Syy-Szz),  Sxy+Syx,        Szx+Sxz;...
    Szx-Sxz,            Sxy+Syx,        (-Sxx+Syy-Szz), Syz+Szy;...
    Sxy-Syx,            Szx+Sxz,        Syz+Szy,        (-Sxx-Syy+Szz) ];


%% Compute eigenvalues of N to get quaternion

[eV,D]=eig(N);

%find max eigenvalue
[~,eigmax]=max(real(diag(D))); 
eigmax = eigmax(1);

if eigmax ~=4
    error('eigmax should be 4 == 4th order polynomial')
end

%use eigenvector matching max eigenvalue
q = real( eV(:,eigmax) ); 

% resolve sign ambiguity
[~,maxQi]=max(abs(q)); 
q=q*sign(q(maxQi(1)));

%normalize using external function
q = qNormalize(q);


%% turning quaternion into rotation matrix using external function
rotMat = qGetR(q);

%% getting scale factor from ratio of sums of rotated data and ref point

rotatedRefSum =0; dataSquaredSum=0;
for i=1:nPts
    rotatedRefSum = rotatedRefSum + dataZeroed(i,:) * rotMat*refZeroed(i,:)';
    dataSquaredSum = dataSquaredSum + dataZeroed(i,:) * dataZeroed(i,:)';
end 

scaleFac = dataSquaredSum/rotatedRefSum; 


%% use rotation matrix to rotate the centroid subtracted reference point.  Any difference left fix can be ascribed to translation

trans = centData' - scaleFac*rotMat*centRef';


%% debug plot to compare rotated, translated and scaled points reference points vs. data

if debugPlots ==1
    
    figure
    set(gcf,'visible','on')
    transformedRefPoints = (scaleFac*rotMat*refPoint' + trans)';
    
    plot3(refPoint(:,1),refPoint(:,2),refPoint(:,3),'color',[0.5,0.5,0.5])
    hold on
    plot3(transformedRefPoints(:,1),transformedRefPoints(:,2),transformedRefPoints(:,3),'r')
    plot3(dataPoint(:,1),dataPoint(:,2),dataPoint(:,3),'b')
    
    axis equal
    title('Rotation+translation+scale.  grey=ref point, red = transformed ref point, blue = data point')
    
end


%% residuals are anything left over
 
residuals =0;
for i=1:nPts
    mismatch = (refPoint(i,:)' - (scaleFac*rotMat*dataPoint(i,:)' + trans));
    residuals = residuals + norm(mismatch);
end      


%% getting axis-angle representation of same roation

if debugPlots ==1
    %for debugging only - getting same info using external library
    %axis-angle using matgeom library. Getting axis-anglef from rotmat to
    %check rotMat is good and matches quaternion info
    R_temp = [rotMat,zeros(3,1); 0, 0, 0, 1];
    [estRotAxis, estRotAngRad] = rotation3dAxisAndAngle(R_temp);
end

%getting axis and angle directly from quaternion
axisAngleRad = [0,0,0, (q(2:4)./norm(q(2:4)))',2*acos(q(1))];

%axis-angle can get to same place in two ways CW theta around positive axis or
%CW 360-theta around negative axis.  Constraining axis to only be
%pointing in front of fly (positive X)
if axisAngleRad(4)<0
   axisAngleRad(4:6) = axisAngleRad(4:6).*-1;
   axisAngleRad(7) = 360-axisAngleRad(7);
end


%% Euler angles
% Euler angles in radians representing a rotation around
% axes in the lab frame, in the following order:
%   1) phi (bank, rotation around x-axis)
%   2) theta (elevation, rotation around y-axis)
%   3) psi (heading, rotation around z-axis)
% This corresponds to a rotation matrix composed as: RzRyRx
%
% Taken from p168 of Quaternions and Rotation Sequences by Jack B. Kuipers

e0 = q(1);
ex = q(2);
ey = q(3);
ez = q(4);
   
ph = atan2( 2*(e0*ex + ey*ez), e0^2+ez^2-ex^2-ey^2 );
th = -asin( 2*(ex*ez-e0*ey) );
ps = atan2( 2*(e0*ez + ex*ey), e0^2+ex^2-ey^2-ez^2 );
    

phi = ph;
theta = th;
psi = ps;

EulerRPY = [phi,theta,psi];
