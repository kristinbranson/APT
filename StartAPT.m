function [labeler, labelerController] = StartAPT(varargin)
% Launch APT, the Advanced Part Tracker. 

% Note that varargin may be coming from the command line in a deployed
% setting.
APT.setpathsmart();
if isscalar(varargin) ,
  args = horzcat({'projfile'}, varargin) ;
else
  args = varargin ;
end
labelerController = LabelerController(args{:}) ;
labeler = labelerController.labeler_ ;
