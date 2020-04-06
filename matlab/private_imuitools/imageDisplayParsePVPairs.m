function [common_args,specific_args] = imageDisplayParsePVPairs(varargin)
%imageDisplayParsePVPairs Pure input parser for image display functions.
%   [common_args,specific_args] =
%   imageDisplayParsePVPairs(specificNames,varargin) parse inputs for image display
%   functions including properties specified in property value pairs. Client-specific
%   arguments specified in specificNames are returned in specific_args.
%
%   common_args is a structure containing arguments that are shared by imtool and imshow.
%
%   specific_args is a structure containing arguments that are specific to a
%   particular client. Arguments specified in specific_args will be returned
%   as fields of specific_args if they are given as p/v pairs by clients.

%   Copyright 2008-2010 The MathWorks, Inc.
%   $Revision: 1.1.6.6 $ $Date: 2011/08/09 17:55:26 $

specific_arg_names = varargin{1};
varargin = varargin(2:end);

% Initialize common_args to default values.
common_args = struct('Filename','',...
    'CData', [],...
    'CDataMapping','',...
    'DisplayRange',[],...
    'Colormap',[],...
    'Map',[],...
    'XData',[],...
    'YData',[],...
    'InitialMagnification',[]);

specific_args = struct([]);

num_args = length(varargin);
params_to_parse = false;

% See if there are parameter-value pairs
% DISPLAY_FUNCTION(...,'Colormap', CMAP,...
%                   'DisplayRange', RANGE,...
%                   'InitialMagnification', INITIAL_MAG,...
%                   'XData', X, ...
%                   'YData', Y,...)
string_indices = find(cellfun('isclass',varargin,'char'));
valid_params = {'Colormap','DisplayRange','InitialMagnification',...
    'XData','YData'};
valid_params = [valid_params,specific_arg_names];

if ~isempty(string_indices) && num_args > 1
    params_to_parse = true;

    is_first_string_first_arg = (string_indices(1)==1);

    nargs_after_first_string = num_args - string_indices(1);
    even_nargs_after_first_string = ~mod(nargs_after_first_string,2);

    if is_first_string_first_arg && even_nargs_after_first_string
        % DISPLAY_FUNCTION(FILENAME,PARAM,VALUE,...)
        param1_index = string_indices(2);
    else
        % DISPLAY_FUNCTION(PARAM,VALUE,...)
        % DISPLAY_FUNCTION(...,PARAM,VALUE,...)
        param1_index = string_indices(1);

        % Make sure first string is a real parameter, if not
        % error here with generic message because user could
        % be trying DISPLAY_FUNCTION(FILENAME,[]).
        matches = strncmpi(varargin{param1_index},valid_params,...
            length(varargin{param1_index}));
        if ~any(matches) %i.e. none
            error(message('images:imageDisplayParsePVPairs:invalidInputs'))
        end
    end
end

if params_to_parse
    num_pre_param_args = param1_index-1;
    num_args = num_pre_param_args;
end

if num_args<0
   error(message('images:validate:tooFewInputs',mfilename));
elseif num_args>2
    error(message('images:validate:tooManyInputs',mfilename));
end
    
    
switch num_args
    case 1
        % DISPLAY_FUNCTION(FILENAME)
        % DISPLAY_FUNCTION(I)
        % DISPLAY_FUNCTION(RGB)

        if (ischar(varargin{1}))
            % DISPLAY_FUNCTION(FILENAME)
            common_args.Filename = varargin{1};
        else
            % DISPLAY_FUNCTION(I)
            % DISPLAY_FUNCTION(RGB)
            common_args.CData = varargin{1};
        end

    case 2
        % DISPLAY_FUNCTION(I,[])
        % DISPLAY_FUNCTION(I,[a b])
        % DISPLAY_FUNCTION(X,map)

        common_args.CData = varargin{1};

        if (isempty(varargin{2}))
            % DISPLAY_FUNCTION(I,[])
            common_args.DisplayRange = 'auto';

        elseif isequal(numel(varargin{2}),2)
            % DISPLAY_FUNCTION(I,[a b])
            common_args.DisplayRange = varargin{2};

        elseif (size(varargin{2},2) == 3)
            % DISPLAY_FUNCTION(X,map)
            common_args.Map = varargin{2};

        else
            error(message('images:imageDisplayParsePVPairs:invalidInputs'))

        end

end

if params_to_parse
    [common_args,specific_args] = parseParamValuePairs(varargin(param1_index:end),valid_params,...
                                                      specific_arg_names,...
                                                      num_pre_param_args,...
                                                      mfilename,...
                                                      common_args,...
                                                      specific_args);
end

%--------------------------------------------------------------------------
function [common_args,specific_args] = parseParamValuePairs(in,valid_params,...
                                                  specific_arg_names,...
                                                  num_pre_param_args,...
                                                  function_name,...
                                                  common_args,...
                                                  specific_args)

if rem(length(in),2)~=0
    error(message('images:imageDisplayParsePVPairs:oddNumberArgs', upper( function_name )));
end

for k = 1:2:length(in)
    prop_string = validatestring(in{k}, valid_params, function_name,...
        'PARAM', num_pre_param_args + k);

    switch prop_string
        case 'DisplayRange'
            if isempty(in{k+1})
                common_args.(prop_string) = 'auto';
            else
                common_args.(prop_string) = checkDisplayRange(in{k+1},mfilename);
            end

        case {'InitialMagnification','Colormap'}
            common_args.(prop_string) = in{k+1};

        case {'XData','YData'}
            common_args.(prop_string) = in{k+1};

        case specific_arg_names
            % A subscript is necessary because specific_args is initialized
            % as an empty struct.
            specific_args(1).(prop_string) = in{k+1};

        otherwise
            error(message('images:imageDisplayParsePVPairs:unrecognizedParameter', prop_string, function_name));

    end
end
