function [ params_x, params_y, params_z ] = dlt_2D_to_3D( DLT, u, v )
% DLT_2D_TO_3D: get equation for line in real wolrd coords given DLT
% parameters and image pt coords.  (Note, not vectorized)
%
% Usage:
%
%   [ params_x, params_y, params_z ] = dlt_2D_to_3D( DLT, u, v )
%
% Inputs:
%
%   DLT = set of 11 DLT coefficients
%
%   u = 1st coord of point in image plane
%
%   v = 2nd coord of point in image plane
%
% Outputs:
%
%   params_x = line parameters of real world x-coord of form 
%              x(t) = param_x(1) + param_x(2)*t
%
%   params_y = line parameters of real world y-coord of form 
%              y(t) = param_y(1) + param_y(2)*t
%
%   params_z = line parameters of real world x-coord of form 
%              z(t) = param_z(1) + param_z(2)*t
%
% 
% Solultion Method:
%
% Solves the DLT equations in the following manner.
%
% DLT equations:   
%
% u = ( DLT(1)*x + DLT(2)*y + DLT(3)*z + DLT(4) )/( DLT(9)*x + DLT(10)*y + DLT(11)*z + 1 )
%
% v = ( DLT(5)*x + DLT(6)*y + DLT(7)*z + DLT(8) )/( DLT(9)*x + DLT(10)*y + DLT(11)*z + 1 )
%
% Linear system formed for (y, z) parametrized by x, in terms of the u, v, DLT(i)
%
% A_x*( y,z)^T = b__x_1 + x*b__x_2
%
% Similarly for (x, z) parameterized by y, and for (x, y) parameterized by z
%
% A_y*(x,z)^T = b_y_1 + y*b_y_2
%
% A_z*(x,y)^T = b_z_1 + z*b_z_2
%
% The system with the lowest condition number is then solved for the desired parameters. 


% matrix for x parametrization  
A_x = [ ...
        u*DLT(10) - DLT(2), u*DLT(11) - DLT(3); ...
        v*DLT(10) - DLT(6), v*DLT(11) - DLT(7)  ...
    ];

b_x_1 = [ -u + DLT(4); -v + DLT(8) ] ;

b_x_2 = [ -u*DLT(9) + DLT(1); -v*DLT(9) + DLT(5) ];

% matrix for y parametrization 

A_y = [ ...
        u*DLT(9) - DLT(1), u*DLT(11) - DLT(3); ...
        v*DLT(9) - DLT(5), v*DLT(11) - DLT(7)  ...
    ];

b_y_1 = [ -u + DLT(4); -v + DLT(8) ];

b_y_2 = [ -u*DLT(10) + DLT(2); -v*DLT(10) + DLT(6) ];

% matrix for z parametrization 

A_z = [ ...
        u*DLT(9) - DLT(1), u*DLT(10) - DLT(2); ...
        v*DLT(9) - DLT(5), v*DLT(10) - DLT(6)  ...
    ];

b_z_1 = [ -u + DLT(4); -v + DLT(8) ];

b_z_2 = [ -u*DLT(11) + DLT(3); -v*DLT(11) + DLT(7) ];

% Get condition numbers

x_cond = cond( A_x );

y_cond = cond( A_y );

z_cond = cond( A_z );

[ temp max_ind ] = min( [ x_cond, y_cond, z_cond ] );

switch max_ind
    
    case 1  % Solve x parameterization 
        
        %disp( '1' )
        
        A_x_inv = inv(A_x);
        
        params_1 = A_x_inv*b_x_1;  
        
        params_2 = A_x_inv*b_x_2;
        
        params_x = [ 0 1 ];   % of form [ b a ] where b + a*t  
        
        params_y = [ params_1(1) params_2(1) ];
        
        params_z = [ params_1(2) params_2(2) ];
        
    case 2 % Solve y parameterization 
        
        %disp( '2' )
        
        A_y_inv = inv(A_y);
        
        params_1 = A_y_inv*b_y_1;  
        
        params_2 = A_y_inv*b_y_2;
        
        params_x = [ params_1(1) params_2(1) ];
        
        params_y = [ 0 1 ];   % of form [ b a ] where b + a*t  
        
        params_z = [ params_1(2) params_2(2) ];
        
    case 3 % Solve z parameterization 
        
        %disp( '3' )
        
        A_z_inv = inv(A_z);
        
        params_1 = A_z_inv*b_z_1;  
        
        params_2 = A_z_inv*b_z_2;
        
        params_x = [ params_1(1) params_2(1) ];
        
        params_y = [ params_1(2) params_2(2) ];
        
        params_z = [ 0 1 ];   % of form [ b a ] where b + a*t  
        
    otherwise
        
        error( 'Fubar we should not be here!' )
        
end

    











