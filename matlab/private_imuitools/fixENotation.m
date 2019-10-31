function string = fixENotation(string)
%fixENotation Make string with E notation look similar on all platforms

%   Copyright 2005 The MathWorks, Inc.
%   $Revision: 1.1.6.1 $  $Date: 2005/12/12 23:22:28 $

if ispc
    string = strrep(string,'E+0','E+');
    string = strrep(string,'E-0','E-');

end
