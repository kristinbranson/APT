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

% Create the LabelerController, which will create the Labeler
labelerController = LabelerController(args{:}) ;

% Extract the Labeler from the LabelerController
labeler = labelerController.labeler_ ;

end  % function
