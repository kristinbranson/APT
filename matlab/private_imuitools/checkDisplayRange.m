function display_range = checkDisplayRange(display_range,fcnName)
%checkDisplayRange display range check function

%   Copyright 2006-2010 The MathWorks, Inc.  
%   $Revision: 1.1.6.6 $  $Date: 2011/08/09 17:55:19 $

if isempty(display_range)
    return
end

validateattributes(display_range, {'numeric'},...
              {'real' 'nonsparse' 'vector','nonnan'}, ...
              fcnName, '[LOW HIGH]', 2);
          
if numel(display_range) ~= 2
    error(message('images:checkDisplayRange:not2ElementVector'))
end

if display_range(2) <= display_range(1)
  error(message('images:checkDisplayRange:badDisplayRangeValues'))
end

display_range = double(display_range);
