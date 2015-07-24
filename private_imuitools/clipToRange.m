function x_clipped = clipToRange(x_candidate, x_range)
%clipToRange Clip point within specified range.
%   x_clipped = clipToRange(x_candidate, x_range) constrains a
%   scalar, x_candidate, to lie within the 1-D range specified by a
%   boundary vector.  The position vector has the form [x_lower x_upper].  
%   The output argument x_clipped is the clipped or constrained value.
%
%   clipToRange assumes the input arguments are correctly formed and does 
%   no error checking on them.

%   $Revision $  $Date: 2005/11/15 01:03:30 $
%   Copyright 2005 The MathWorks, Inc.

    x_clipped = min( x_range(2), max( x_candidate, x_range(1) ) );