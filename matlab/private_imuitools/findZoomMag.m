function y = findZoomMag(dir,x)
%findZoomMag Find magnification for zooming in or out.
%    Y = findZoomMag(DIR,X)
%    
%    DIR is 'in' or 'out' for zooming in or zooming out, respectively.
%    
%    Round X up/down to the next element in the sequence:
%    ..., 1/4, 1/3, 1/2, 2/3, 1, 2, 4, 8, ...
%    
%    X and Y are in natural units. Multiply by 100 to convert to
%    percentage. 
%    
%    For example, if X = 2, that's equivalent to doubling the magnification,
%    also described as 200% magnification. And, if X = .6, that's equivalent
%    to reducing the magnification by 60%.

%   Copyright 2004-2006 The MathWorks, Inc.  
%   $Revision: 1.1.8.4 $  $Date: 2007/01/15 14:37:56 $

zoomingIn = strcmp(dir,'in');

log2_x = log(x) / log(2.);

% find next integer powers of 2 even if you're on an integer power of 2 
power_of_2_larger  = 2^(floor(log2_x) + 1);
power_of_2_smaller = 2^(ceil( log2_x) - 1);

% If we're effectively on an integer power of 2 as defined by magPercentsDiffer,
% correct as needed by bumping the corresponding power of 2 up or down.
if ~magPercentsDiffer(x,power_of_2_larger)
    power_of_2_larger = 2*power_of_2_larger;
elseif ~magPercentsDiffer(x,power_of_2_smaller)
    power_of_2_smaller = power_of_2_smaller/2;
end

if ~magPercentsDiffer(x,1)
    if zoomingIn
        y = 2;
    else
        y = 2/3;
    end
    
elseif (x > 1) 

    % Stay on integer power of 2    

    if zoomingIn
        y = power_of_2_larger;        
    else
        y = power_of_2_smaller;                    
    end

else % (x < 1)

    % Be finer grained than integer powers of 2 (1, 1/2, 1/4, ...). 
    %
    % So "in between" each pair of power of two mags, we find another mag.
    
    if zoomingIn
        y = findZoomingInMag(x,power_of_2_larger);
        
    else
        y = findZoomingOutMag(x,power_of_2_smaller);
        
    end
        
end

%--------------------------------------------------------------------------
function y = findZoomingInMag(x,power_of_2_larger)

in_between_mag = power_of_2_larger * 2/3;
        
if (x < in_between_mag) && magPercentsDiffer(x,in_between_mag);
    y = in_between_mag;

else
    y = power_of_2_larger;
    
end

%--------------------------------------------------------------------------
function y = findZoomingOutMag(x,power_of_2_smaller)

in_between_mag = power_of_2_smaller * 4/3;        

if (x > in_between_mag) && magPercentsDiffer(x,in_between_mag);
    y = in_between_mag;

else
    y = power_of_2_smaller;
    
end

