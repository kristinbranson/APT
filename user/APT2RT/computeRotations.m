function [q,trans,scaleFac,residuals,rotMat,axisAngleRad,EulerRPY]=computeRotations(refPoint,dataPoint)
%estimates rotation & translation between two clouds of points.  Returns
%rotation information in a varienty of formats.
%
% refPoint/dataPoint format = nPts x 3.  columns = xyz, rows = points.
%
% q = quaternion describing rotation.
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
    tempM = mean(dataZeroed,1)
    plot3(tempM(1),tempM(2),tempM(3),'ro')
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

%normalize
q = q(:);
q = q/norm(q);


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



%% residuals are anything left over
 
residuals =0;
for i=1:nPts
    mismatch = (refPoint(i,:)' - (scaleFac*rotMat*dataPoint(i,:)' + trans));
    residuals = residuals + norm(mismatch);
end      


%% getting axis-angle representation of same roation

%axis-angle using matgeom library
R_temp = [rotMat,zeros(3,1); 0, 0, 0, 1];
[estRotAxis, estRotAngRad] = rotation3dAxisAndAngle(R_temp);


axisAngleRad = [estRotAxis,estRotAngRad];

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
