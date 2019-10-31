function loadJavaCustomizations()
% loadJavaCustomizations - Load the custom java files
% -------------------------------------------------------------------------
% Abstract: Loads the custom Java .jar file required for the
% uiextras.jTable
%
% Syntax:
%           loadJavaCustomizations()
%
% Inputs:
%           none
%
% Outputs:
%           none
%
% Examples:
%           none
%
% Notes: none
%

%   Copyright 2012-2015 The MathWorks, Inc.
%
% Auth/Revision:
%   MathWorks Consulting
%   $Author: rjackey $
%   $Revision: 1078 $  $Date: 2015-02-20 09:13:35 -0500 (Fri, 20 Feb 2015) $
% ---------------------------------------------------------------------

% Define the jar file
JarFile = 'UIExtrasTable.jar';
JarPath = fullfile(fileparts(mfilename('fullpath')), JarFile);

% Check if the jar is loaded
JavaInMem = javaclasspath('-all');
PathIsLoaded = ~all(cellfun(@isempty,strfind(JavaInMem,JarFile)));

% Load the .jar file
if ~PathIsLoaded
    disp('Loading Java Customizations in UIExtrasTable.jar');
    javaaddpath(JarPath);
end
