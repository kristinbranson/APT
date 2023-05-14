function [labeler, labelerController] = StartAPT(varargin)
APT.setpathsmart;
if isscalar(varargin) ,
  args = horzcat({'projfile'}, varargin) ;
else
  args = varargin ;
end
labelerController = LabelerController(args{:}) ;
labeler = labelerController.labeler_ ;
