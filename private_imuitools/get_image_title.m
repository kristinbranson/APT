function label = get_image_title(arg,arg_name)
%GET_IMAGE_TITLE gets title of image from its string name or variable name
%
%   LABEL = GET_IMAGE_TITLE(arg,arg_name) returns a string containing the
%   arg's name.
%
%   Class Support
%   -------------
%   ARG can be a string or a variable name. 
%
%   Examples
%   --------
%   INPUTNAME is a function that has to be called within the body of a
%   user-defined function. This line would be within a parse_inputs
%   subfunction in a toolbox function.
%
%   inputImageName = get_image_title(varargin{1},inputname(1));

%   Copyright 1993-2008 The MathWorks, Inc.
%   $Revision: 1.1.6.2 $  $Date: 2008/04/03 03:12:49 $

if ischar(arg)
    [path, label] = fileparts(arg);
else 
    label = arg_name;
end


