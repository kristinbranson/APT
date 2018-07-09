function [alignedheadCoords, alignedBodyCoords, alignedPivotPoint] = alignBodyAxis2Xaxis(headCoords,bodyCoords,pivotPoint)
%%  Function to put data in reference frame where body axis is X-axis (positive forward),
% z-axis is up, Y axis is fly's left(positive) and head pivot point is origin
%
% Aligns data so wingbase-wingbase line is aligned with Y axis (postive = left), body axis
% is aligned with X-axis, z-axis is up and origin is at your estimate of
% the head-pivot point (not always aligned with the body axis)
%
% bodyCoords input format = [neck; thorax-abdomen joint; left wing base; right wingbase]
% format of bodyCoords = ...
% [   bodyData.kine.flyhead.data.coords(i_neckCentre,:,bodyAngleFrame);...  
%     bodyData.kine.flyhead.data.coords(i_thoracicAbdomenJoint,:,bodyAngleFrame);... 
%     bodyData.kine.flyhead.data.coords(i_LwingBase,:,bodyAngleFrame);... 
%     bodyData.kine.flyhead.data.coords(i_RwingBase,:,bodyAngleFrame)];




%getting distance ratios between body landmarks so can make body model
%aligned with y-axis
n2a = norm(bodyCoords(1,:)-bodyCoords(2,:));%neck to thorax-abdomen joint distance
b2b = norm(bodyCoords(3,:)-bodyCoords(4,:));%wing base to wing base distance.  Left Wing base - Right wing base

%projecting mid-point between two wing bases onto ventral body axis so can
%find distance along body axis from neck to wing base centroid
baseMiddleOnBodyAxis = projPointOnLine3d(mean(bodyCoords(3,:),1), [bodyCoords(1,:),bodyCoords(2,:)-bodyCoords(1,:)]);
n2b_X=norm(bodyCoords(1,:)-baseMiddleOnBodyAxis); %distance from neck to centroid of wing bases just along ventral body axis

%vertical distance from ventral body axis to height of wing bases
ba2b_Z = distancePointLine3d(mean(bodyCoords(3:4,:),1), [bodyCoords(1,:),bodyCoords(2,:)-bodyCoords(1,:)]);%vertical distance from ventral body axis to height of wing bases

%scaling distances so neck-to-throaxabdomen joint body length is 1 - just helps
%debugging
b2b = b2b/n2a;
n2b_X = n2b_X/n2a;
ba2b_Z = ba2b_Z/n2a;
n2a = n2a/n2a;


%making fake body axis aligned with calibration coordinate system
%body model is thorax length with distance between wing hinges.  These two
%lengths are roughly the same and wings fall roughly half way down thorax
%length. Wing bases are roughly 0.5 thorax length above thorax axis line that lies along bottom side of fly (what you should digitize in kine).
%Making basic model like this with thorax of unit length down X
%axis and wing base to wing base line of unit length down Y axis and neck
%at origin

            % X,          Y,        Z
bodyModel = [ 0,         0,        0;...%neck
             -1*n2a,     0,        0;...%thorax-abdomen joint on ventral side
             -1*n2b_X,   b2b/2,    ba2b_Z;...%left wing base
             -1*n2b_X,   b2b/-2,   ba2b_Z]; %right wing base

         
%get rotation and translation matrix and scale factor necessary to put body axis in
%alignment with bodyModel and thus calibration coordinate system
[D2M_scale, D2M_R, D2M_T, D2M_residuals] = absoluteOrientationQuaternion(bodyCoords', bodyModel');

%using estimated rotation, translation matrices and scale factor to rotate
%all data into same coord system as calibration - so that body axis lies
%along y axis
alignedBodyCoords = ( D2M_scale*D2M_R*bodyCoords'+ repmat(D2M_T,1,size(bodyCoords,1)) )';
alignedheadCoords = ( D2M_scale*D2M_R*headCoords'+ repmat(D2M_T,1,size(headCoords,1)) )';
alignedPivotPoint = ( D2M_scale*D2M_R*pivotPoint'+ D2M_T )';

% zeroing all coordinates so pivot point is origin
alignedheadCoords = alignedheadCoords - repmat(alignedPivotPoint,size(alignedheadCoords,1),1);
alignedBodyCoords = alignedBodyCoords - repmat(alignedPivotPoint,size(alignedBodyCoords,1),1);
bodyModel = bodyModel - repmat(alignedPivotPoint,size(bodyModel,1),1);
alignedPivotPoint = [0,0,0];

% figure
% plot3(bodyModel(:,1),bodyModel(:,2),bodyModel(:,3),'k')
% hold on
% plot3(alignedBodyCoords(:,1),alignedBodyCoords(:,2),alignedBodyCoords(:,3),'b')
% axis equal
% xlabel('x')
% ylabel('y')
% zlabel('z')


