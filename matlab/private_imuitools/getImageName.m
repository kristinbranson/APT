function imageName = getImageName(filename,inputVariableName)
%getImageName Gets name of image from its string name or variable name.
%   imageName = getImageName(FILENAME,inputVariableName) returns a string
%   containing the image name.
%
%   Examples
%   --------
%   INPUTNAME is a function that has to be called within the body of a
%   user-defined function. This line would be within a function calling
%   getImageName:
%
%      inputImageName = getImageName(varargin{1},inputname(1));

%   Copyright 2005 The MathWorks, Inc.
%   $Revision $  $Date: 2005/05/27 14:07:18 $

if ~ischar(filename) || isempty(filename)
  if isempty(inputVariableName)
    imageName = '(MATLAB Expression)';
  else
    imageName = inputVariableName;
  end
else
  [dummy,name,ext] = fileparts(filename); %#ok dummy unused
  imageName = sprintf('%s%s',name,ext);
end
