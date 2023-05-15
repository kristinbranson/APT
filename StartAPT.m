function [labeler, labeler_controller] = StartAPT(varargin)
APT.setpathsmart;
if isscalar(varargin) ,
  args = horzcat({'projfile'}, varargin) ;
else
  args = varargin ;
end
labeler_controller = labeler_controller_object(args{:}) ;
labeler = labeler_controller.labeler_ ;
