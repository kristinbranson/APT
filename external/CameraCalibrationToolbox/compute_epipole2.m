function [epipole] = compute_epipole2(xLp,R,T,...
  fc_right,cc_right,kc_right,alpha_c_right,...
  fc_left,cc_left,kc_left,alpha_c_left,varargin)
% xLp, cc_right, cc_left: 0b per caltech toolbox conventions
%
% epipole: 0b etc

[roi,roifac,roiradius,N_line] = myparse(varargin,...
  'roi',[], ... % (opt) [xlo xhi ylo yhi] ROI (in right cam) in which to compute EP line (approx)
  'roifac',0.3, ... % fudge factor to expand roi; used with 'roi'
  'roiradius',400, ... % (opt) radius of square roi in px; used if 'roi' not supplied
  'N_line',800 ... % number of points to compute in EPline
  );

tfROI = ~isempty(roi);

% xL: 3d coords for xLp (at z=1), in left camera coord sys
xL = [ normalize_pixel(xLp,fc_left,cc_left,kc_left,alpha_c_left); 1 ];

S = [   0  -T(3)  T(2)
    T(3)   0   -T(1)
    -T(2)  T(1)   0 ];

% xR = R*xL + T
% T x xR = T x R*xL + 0
%        = S*R*xL
%
% - T is the vector pointing from the camR origin to the camL origin, in
% camR's coord sys.
% - xR is the vector pointing to xR, in camR's coord sys
% - both lie in the EP plane => T x xR is normal to the EP plane

nEP = (S*R)*xL; % normal to the EP plane (not a unit vec)

% l_epipole x q lies in the EP plane for any q

if norm(nEP(2)) > norm(nEP(1)),
    % Horizontal(ish) EP plane/line:
    
    % We want to draw the EPline across the desired range/roi
    %
    % if (u,v) are image coords in camR and
    % if (x,y,z) are 3d coords in camR coord sys then
    % x/z = (u - cc_x) / fc_x
    %
    % In the following we compute limit_x_neg/pos as extreme values of x/z

    if tfROI
      droi = roifac*(roi(2)-roi(1));
      assert(droi>0);
      xlo = roi(1)-droi/2;
      xhi = roi(2)+droi/2;
    else
      % appears to assume similar viewpoints for camL / camR as this 
      % centers the camR roi around xLp 
      xlo = xLp(1) - roiradius/2;
      xhi = xLp(1) + roiradius/2;
    end
    limit_x_neg = (xlo - cc_right(1)) / fc_right(1);
    limit_x_pos = (xhi - cc_right(1)) / fc_right(1);    
    x_list = (limit_x_pos - limit_x_neg) * ((0:(N_line-1)) / (N_line-1)) + limit_x_neg;

    % Consider l_epipole x q
    % - For a very horizontal EP plane/line, nEP(2) >> nEP(1)
    % - Meanwhile we will use q(1)=1 and q(2)=0 identically (see below)
    % => nEP x q ~ ( nEP(2)*q(3), ~, -nEP(2) )
    % normalizing by the z-coord
    % => nEP x q ~ ( -q(3), ~, 1 )
    %
    % By setting q(3) = -x_list, we are asking for an EP line that extends
    % across the extremal x/z as desired
    pt = cross(repmat(nEP,1,N_line),[ones(1,N_line);zeros(1,N_line);-x_list]);
else
    % Vertical(ish) EP plane/line:

    if tfROI
      droi = roifac*(roi(4)-roi(3));
      assert(droi>0);
      ylo = roi(3)-droi/2;
      yhi = roi(4)+droi/2;
    else
      % appears to assume similar viewpoints for camL / camR as this 
      % centers the camR roi around xLp 
      ylo = xLp(2) - roiradius/2;
      yhi = xLp(2) + roiradius/2;
    end
    limit_y_neg = (ylo - cc_right(2)) / fc_right(2);
    limit_y_pos = (yhi - cc_right(2)) / fc_right(2);
    y_list = (limit_y_pos - limit_y_neg) * ((0:(N_line-1)) / (N_line-1)) + limit_y_neg;
    
    % Consider l_epipole x q
    % - For a very vertical EP plane/line, nEP(1) >> nEP(2)
    % - Meanwhile we will use q(1)=0 and q(2)=1 identically (see below)
    % => nEP x q ~ ( ~, -nEP(1)*q(3), nEP(1) )
    % normalizing by the z-coord
    % => nEP x q ~ ( ~, -q(3), 1 )
    %
    % By setting q(3) = -y_list, etc
    pt = cross(repmat(nEP,1,N_line),[zeros(1,N_line);ones(1,N_line);-y_list]); 
end

pt = [pt(1,:) ./ pt(3,:) ; pt(2,:)./pt(3,:)];
ptd = apply_distortion(pt,kc_right);
KK_right = [fc_right(1) alpha_c_right*fc_right(1) cc_right(1); ...
            0 fc_right(2) cc_right(2); ...
            0 0 1];
epipole = KK_right * [ ptd ; ones(1,N_line)];
epipole = epipole(1:2,:);