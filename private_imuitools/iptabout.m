function iptabout(varargin)
%IPTABOUT About the Image Processing Toolbox.
%   IPTABOUT displays the version number of the Image Processing
%   Toolbox and the copyright notice in a modal dialog box.

%   Copyright 1993-2008 The MathWorks, Inc.
%   $Revision: 1.1.8.3 $ $Date: 2008/06/20 07:59:54 $ 

tlbx = ver('images');
tlbx = tlbx(1);
str = sprintf('%s %s\nCopyright 1993-%s The MathWorks, Inc.', ...
              tlbx.Name, tlbx.Version, datestr(tlbx.Date, 10));
s = load(fullfile(ipticondir, 'iptabout.mat'));
num_icons = numel(s.icons);
stream = RandStream('mcg16807','seed',sum(100*clock));
icon_idx = randi(stream, num_icons); % random integer in 1:num_icons
msgbox(str,tlbx.Name,'custom',s.icons{icon_idx},gray(64),'modal');

