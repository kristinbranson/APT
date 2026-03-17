function [labeler, labelerController] = StartAPT(varargin)
% Launch APT, the Advanced Part Tracker. 

% Note that varargin may be coming from the command line in a deployed
% setting.

% Set up the APT path
APT.setpathsmart();

% If only one argument, assume it's a project file to be opened
if isscalar(varargin) ,
  args = horzcat({'projfile'}, varargin) ;
else
  args = varargin ;
end

% Process args that have to be dealt with before creating the Labeler
[isInDebugMode, isInAwsDebugMode, args] = ...
  myparse_nocheck(args, ...
                  'isInDebugMode',false, ...
                  'isInAwsDebugMode',false) ;

% Create the labeler, tell it there will be a GUI attached
labeler = Labeler('isgui', true, 'isInDebugMode', isInDebugMode,  'isInAwsDebugMode', isInAwsDebugMode) ;  

% Create the LabelerController
labelerController = LabelerController(labeler, args{:}) ;

% % Extract the Labeler from the LabelerController
% labeler = labelerController.labeler_ ;

% If a projfile was given, load it
[projfile, replace_path] = ...
  myparse_nocheck(varargin, ...
                  'projfile',[], ...
                  'replace_path',{'',''}) ;
if projfile ,
  labeler.projLoad(projfile, 'replace_path', replace_path) ;
end      

end  % function
