function state = logical2onoff(b)
%LOGICAL2ONOFF Translate logical into 'on' or 'off' string.
%   STATE = LOGICAL2ONOFF(B) where B is a logical scalar returns the string 'on'
%   if B is true and the string 'off' if B is false.
%
%   This function is useful for setting HG properties without writing a lot of
%   conditional code.

%   Copyright 2005 The MathWorks, Inc.
%   $Revision $ $Date: 2005/05/27 14:07:29 $
  
  if b
    state = 'on';
  else
    state = 'off';
  end
  
end
