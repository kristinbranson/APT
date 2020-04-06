function initializeImformatsFileFilter
%INITIALIZEIMFORMATSFILEFILTER
%   INITIALIZEIMFORMATSFILEFILTER initializes the file format descriptions
%   and extensions available to instances of ImformatsFileFilter
%   
%   See also IMFORMATS, IMPUTFILE, IMGETFILE.

%   Copyright 2007-2008 The MathWorks, Inc.
%   $Revision: 1.1.6.2 $  $Date: 2008/11/24 14:58:42 $
    
import com.mathworks.toolbox.images.ImformatsFileFilter;

% Parse formats from IMFORMATS (plus DICOM)
[desc ext] = iptui.parseImageFormats;
nformats = length(desc);

% Create a filter that includes all extensions
ext_all = cell(0);
for i = 1:nformats
    ext_i = ext{i};
    ext_all(end+1: end+numel(ext_i)) = ext_i(:);
end
ext{end+1,1}  = ext_all;
desc{end+1,1} = 'All image files';

% Make a vector of String arrays (java is zero based)
extVector = java.util.Vector(nformats);
for i = 0:nformats
    extVector.add(i,ext{i+1})
end

% Push formats into ImformatsFileFilter so instances of
% ImformatsFileFilter will be based on IMFORMATS.
ImformatsFileFilter.initializeFormats(nformats,desc,extVector);
