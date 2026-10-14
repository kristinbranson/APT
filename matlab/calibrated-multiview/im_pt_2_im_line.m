function [u_2, v_2] = im_pt_2_im_line( u_1, v_1, DLT_1, DLT_2, im_bndry, len )
% IM_PT_2_IM_LINE: get line in image #2 corresponding to point (u_1, v_1)
% in image #1
%
% DLT_1 = 11 DLT coefficients for camera 1
% DLT_2 = 11 DLT coefficients for camera 2
% im_bndry = [xmin xmax ymin ymax] for image #2
% len = number of points generated along line

[ params_x, params_y, params_z ] = dlt_2D_to_3D( DLT_1, u_1, v_1 );

% Find line intersections with image bndry

s_list = [];

% intersection with min( u_2 ) and max( u_2 )
[ numer_min, denom_min ] = u_intersect_func(  params_x, params_y, params_z, DLT_2, im_bndry(1)  );

[ numer_max, denom_max ] = u_intersect_func(  params_x, params_y, params_z, DLT_2, im_bndry(2)  );

if abs(denom_min) > abs(eps) & abs(denom_max) > abs(eps) 

    s_list = [ s_list,  numer_min/denom_min, numer_max/denom_max ];
    
end

% intersection with min( v_2 ) and max( v_2 )
[ numer_min, denom_min ] = v_intersect_func(  params_x, params_y, params_z, DLT_2, im_bndry(3)  );

[ numer_max, denom_max ] = v_intersect_func(  params_x, params_y, params_z, DLT_2, im_bndry(4)  );

if abs(denom_min) > abs(eps) & abs(denom_max) > abs(eps)

    s_list = [ s_list,  numer_min/denom_min, numer_max/denom_max ];
    
end

% No intersection with image bndry
if isempty( s_list )
    
    u_2 = [];
    
    v_2 = [];
   
else
  
  % AL20160712. Note the line generated here may extend beyond the
  % boundaries of im_bndry (when projected onto image 2), apparently 
  % because we take the absolute min and max of the various s parameters   
  % without regard to which boundaries of image 2 are hit by the line
  % "earlier" (inner) vs "later" (outer).
    
    s_max = max( s_list );
    
    s_min = min( s_list );
    
    s = linspace( s_min, s_max, len );
    
    x = params_x(1) + params_x(2)*s;
    
    y = params_y(1) + params_y(2)*s;
    
    z = params_z(1) + params_z(2)*s;
   
    [ u_2, v_2 ] = dlt_3D_to_2D( DLT_2, x, y, z );
    
%     fprintf('u_2 = %s, v_2 = %s\n',mat2str(u_2),mat2str(v_2));
%     fprintf('x = %s, y = %s, z = %s\n',mat2str(x),mat2str(y),mat2str(z));
%     fprintf('u_1 = %s, v_1 = %s\n',mat2str(u_1),mat2str(v_1));
%     fprintf('DLT_1 = %s, DLT_2 = %s\n',mat2str(DLT_1),mat2str(DLT_2));
%     fprintf('s_list = %s\n',mat2str(s_list));
%     fprintf('numer_min = %f, denom_min = %f, numer_max = %f, denom_max = %f\n',...
%       numer_min,denom_min,numer_max,denom_max);
    
end

% -------------------------------------------------------------------------
function [ numer, denom ] = u_intersect_func(  params_x, params_y, params_z, DLT, val )
% INTERSECT_FUNC:  get numerator and denominator of interesction point

% numer = params_x(1)*( -val*DLT(9) + DLT(1) )  + params_y(1)*( -val*DLT(10) + DLT(2) ) + ...
%     params_z(1)*( -val*DLT(11) + DLT(3) ) + DLT(4) - val;
% 
% denom = params_x(2)*(  DLT(1) - val*DLT(9) )  + params_y(2)*( DLT(2) - val*DLT(10)) + ...
%     params_z(2)*( DLT(3) - val*DLT(11) );

numer = params_x(1)*( -val*DLT(9) + DLT(1) )  + params_y(1)*( -val*DLT(10) + DLT(2) ) + ...
    params_z(1)*( -val*DLT(11) + DLT(3) ) + DLT(4) - val;

denom = params_x(2)*(  -DLT(1) + val*DLT(9) )  + params_y(2)*( -DLT(2) + val*DLT(10)) + ...
    params_z(2)*( -DLT(3) + val*DLT(11) );


% -------------------------------------------------------------------------
function [ numer, denom ] = v_intersect_func(  params_x, params_y, params_z, DLT, val )
% INTERSECT_FUNC:  get numerator and denominator of interesction point

% numer = params_x(1)*( -val*DLT(9) + DLT(5) )  + params_y(1)*( -val*DLT(10) + DLT(6) ) + ...
%     params_z(1)*( -val*DLT(11) + DLT(7) ) + DLT(8) - val;
% 
% denom = params_x(2)*(  DLT(5) - val*DLT(9) )  + params_y(2)*( DLT(6) - val*DLT(10)) + ...
%     params_z(2)*( DLT(7) - val*DLT(11) );

numer = params_x(1)*( -val*DLT(9) + DLT(5) )  + params_y(1)*( -val*DLT(10) + DLT(6) ) + ...
    params_z(1)*( -val*DLT(11) + DLT(7) ) + DLT(8) - val;

denom = params_x(2)*(  -DLT(5) + val*DLT(9) )  + params_y(2)*( -DLT(6) + val*DLT(10)) + ...
    params_z(2)*( -DLT(7) + val*DLT(11) );
