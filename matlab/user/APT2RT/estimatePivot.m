function [pivot]= estimatePivot(points,pivot)
% function to estimate pivot/orgin that 3D points rotate about.  Given 3D
% points at multiple time points that rotate about some pivot it will
% iteratively run an optimization to try and find that pivot point.
%
% See kine2RotationWrapper.m for example of how to use this.
%
% Inputs:
% 
%       points = n x 3 x t 3D array.  Each row is a diffrent 3D point, each
%       of 3 columns are the xyz coords of the point, each 3rd dimension is
%       a different time point where the 3D points are rotated relative to
%       other time points about a pivot.  origin of xyz points determines
%       the starting point in the optimization to find the pivot.
%       
%       pivot = 1x3 vector specifying xyz location of users best guess as
%       to pivot point - usually from clicking on fly's neck.
%
%   
%
% Outputs:
%   
%           1x3 vector giving xyz coords of best estimate of pivot point -
%           SUBTRACT this to points to put them in correct coordinate system to
%           apply absoluteOrientationQuaternion() etc.
%
%
%
% Dependencies:
%   http://www.mathworks.com/matlabcentral/fileexchange/22422-absolute-orientation

%    disp('Finding best pivot point, please wait this will take a while...')
        

    %optimizing output of pivotError to find best pivot, starting with [0,0,0]
    %disp(['User supplied pivot = ', num2str(pivot)])
     options = optimset('MaxFunEvals',6000);
    pivot = fminsearch(@(pivot) pivotError(pivot,points), pivot,options); %fminsearch and fminunc produce same output but fminunc is quicker - use whichever you have the toolbox for, both are fine
    %disp(['Estimated pivot = ',num2str(pivot)])

    
end%end of function



function [err] = pivotError(pivot,points);
% subfunction used by estimatePivot.m
% takes 3D points input for points at different timesteps.  Finds the best
% rotation axis originating at current pivot point that can align points.
% Returns the sum of these residuals after rotating all the points to align
% them.  Pretty sure there is a better non-optimiation way to do this.
%
% Inputs:
% 
%       points = n x 3 x t 3D array.  Each row is a diffrent 3D point, each
%       of 3 columns are the xyz coords of the point, each 3rd dimension is
%       a different time point where the 3D points are rotated relative to
%       other time points about a pivot.  origin of xyz points determines
%       the starting point in the optimization to find the pivot
%
%       pivot = 1x3 vector specifying xyz location of pivot point in reference frame/coords
%       of original points data - will be subtracted from points to put
%       them in reference frame of new origin
%
% Outputs:
%
%       err = cumulative error over all points and all frames describing
%       the distance between each 3D point and its equivalent after being
%       rotated about the current pivot point to best align them

%warning('currently just doing comparison between successive frames, need to change to all-to-all')

    inc = 10; % increment to step through frames with - bigger the number, the faster it runs but the more frames you skip

    err=0;
    

    for c=1:inc:size(points,3) %for each comparison starting point
        
        for t=c:inc:size(points,3)% for each time point from current comparison point onwards

            %points to try and align to
            comparisonPoints = points(:,:,c)' - repmat(pivot',1,size(points,1));

            %points at current time point
            points_t = points(:,:,t)' - repmat(pivot',1,size(points,1));


            %find rotation (and translation) about current origin that best aligns two sets of points
            [scale, R, T, residuals]=...
                absoluteOrientationQuaternion(comparisonPoints,points_t);

            %apply JUST rotation to points at t-1 to try and estimate points at t,
            %specify error between the two (cumulative error across all frames tested)
    %        difference = points_t-(scale*R*points_tminus1+ repmat(T,1,size(points_tminus1,2)));
            difference = points_t-(R*comparisonPoints);
            for n=1:size(difference,2) %for each 3D point
                err=err+norm(difference(:,n));
            end

    %         rotcomparisonpoints=(R*comparisonPoints);
    %         plot3(points_t(1,:),points_t(2,:),points_t(3,:),'k')
    %         hold on
    %         plot3(comparisonPoints(1,:),comparisonPoints(2,:),comparisonPoints(3,:),'b')
    %         plot3(rotcomparisonpoints(1,:),rotcomparisonpoints(2,:),rotcomparisonpoints(3,:),'b--')
    %         hold off
    %         drawnow
    %         err
    %         pause


        end
        
    end 

end%end of function