function apt = StartAPT(varargin)
APT.setpathsmart;
if isscalar(varargin) ,
  args = horzcat({'projfile'}, varargin) ;
else
  args = varargin ;
end
apt = Labeler(args{:});
