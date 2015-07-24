function common_args = imageDisplayValidateParams(common_args)
%imageDisplayValidateParams Validate and set defaults of image display
%functions.
%   commonArgs = imageDisplayValidateParams(commonArgs) validate commonArgs
%   structure returned by imageDisplayParsePVPairs. Set default values for
%   unspecified parameters and validate specified parameters.

%   Copyright 2008-2011 The MathWorks, Inc.
%   $Revision: 1.1.6.7 $ $Date: 2011/08/09 17:55:27 $

% Make sure CData is numeric before going any further.
validateattributes(common_args.CData, {'numeric','logical'},...
    {'nonsparse'}, ...
    mfilename, 'I', 1);

if isempty(common_args.XData)
    common_args.XData = [1 size(common_args.CData,2)];
end

if isempty(common_args.YData)
    common_args.YData = [1 size(common_args.CData,1)];
end

% Validate XData YData
checkCoords(common_args.XData,'XDATA');
checkCoords(common_args.YData,'YDATA');

image_type = findImageType(common_args.CData,common_args.Map);

% validate CData and any user supplied Colormap
common_args.CData = validateCData(common_args.CData,image_type);
common_args.Map = validateMap(common_args.Map,common_args.Colormap,...
    image_type);

% we now only need a single 'Map' field, so we remove 'Colormap'
common_args = rmfield(common_args,'Colormap');

common_args.CDataMapping = getCDataMapping(image_type);

if strcmp(common_args.DisplayRange,'auto')
    common_args.DisplayRange = getAutoCLim(common_args.CData,image_type);
end

if strcmp(common_args.CDataMapping,'scaled')
    
    % set colormap if user did not provide one
    if isempty(common_args.Map)
        common_args.Map = gray(256);
    end

    if isempty(common_args.DisplayRange) || ...
            (common_args.DisplayRange(1) == common_args.DisplayRange(2))
        common_args.DisplayRange = getrangefromclass(common_args.CData);
    end
end

common_args.DisplayRange = checkDisplayRange(common_args.DisplayRange,mfilename);

%---------------------------------
function clim = getAutoCLim(cdata,image_type)

% RGB images do not use the CLim, so we set "auto" CLim to the default
% class display range (g731516)
if strcmpi(image_type,'truecolor')
    clim = getrangefromclass(cdata);
else
    clim = double([min(cdata(:)) max(cdata(:))]);
end
        
%----------------------------------------------------
function [cdatamapping] = getCDataMapping(image_type)

cdatamapping = 'direct';

% cdatamapping is not relevant for RGB images, but we set it to something so
% we can call IMAGE with one set of arguments no matter what image type.

% May want to treat binary images as 'direct'-indexed images for display
% in HG which requires no map.
%
% For now, they are treated as 'scaled'-indexed images for display in HG.

switch image_type
    case {'intensity','binary'}
        cdatamapping = 'scaled';

    case 'indexed'
        cdatamapping = 'direct';

end

%-----------------------------------------------
function map = validateMap(map,user_cmap,image_type)

% use user supplied colormap if possible
if ~isempty(user_cmap)
    map = user_cmap;
end

% discard provided maps for truecolor images
if ~isempty(map) && strcmp(image_type,'truecolor');
    warning(message('images:imageDisplayValidateParams:colormapWithTruecolor'));
    map = [];
end

% colormap must be m-by-3 matrix of numeric
if ~isempty(map)
    if ~isequal(ndims(map),2) || ~isequal(size(map,2),3) || ~isnumeric(map)
        error(message('images:imageDisplayValidateParams:invalidColormap'))
    end
end

%-----------------------------------------------
function cdata = validateCData(cdata,image_type)

if ((ndims(cdata) > 3) || ((size(cdata,3) ~= 1) && (size(cdata,3) ~= 3)))
    error(message('images:imageDisplayValidateParams:unsupportedDimension'))
end

if islogical(cdata) && (ndims(cdata) > 2)
    error(message('images:imageDisplayValidateParams:expected2D'));
end

% RGB images can be only be uint8, uint16, single, or double
if ( (ndims(cdata) == 3)   && ...
        ~isa(cdata, 'double') && ...
        ~isa(cdata, 'uint8')  && ...
        ~isa(cdata, 'uint16') && ...
        ~isa(cdata, 'single') )
    error(message('images:imageDisplayValidateParams:invalidRGBClass'))
end

if strcmp(image_type,'indexed') && isa(cdata,'int16')
    error(message('images:imageDisplayValidateParams:invalidIndexedImage'))
end

% Clip double and single RGB images to [0 1] range
if ndims(cdata) == 3 && ( isa(cdata, 'double') || isa(cdata,'single') )
    cdata(cdata > 1) = 1;
    cdata(cdata < 0) = 0;
end

% Catch complex CData case
if (~isreal(cdata))
    warning(message('images:imageDisplayValidateParams:displayingRealPart'))
    cdata = real(cdata);
end

%----------------------------------------
function checkCoords(coords,coord_string)

validateattributes(coords, {'numeric'}, {'real' 'nonsparse' 'finite' 'vector'}, ...
    mfilename, coord_string, []);

if numel(coords) < 2
    error(message('images:imageDisplayValidateParams:need2Coords', coord_string));
end

%----------------------------------------
function imgtype = findImageType(img,map)

if (isempty(map))
    if ndims(img) == 3
        imgtype = 'truecolor';
    elseif islogical(img)
        imgtype = 'binary';
    else
        imgtype = 'intensity';
    end
else
    imgtype = 'indexed';
end
