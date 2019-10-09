function [common_args,specific_args] = roiParseInputs(low,high,varargin,client_name,specific_arg_names)
%roiParseInputs Parse inputs of ROI tools.  
%    [commonArgs,specific_args] =
%    roiParseInputs(LOW,HIGH,VARARGIN,CLIENT_NAME,SPECIFIC_ARG_NAMES) parses
%    the input of an ROI tool. LOW and HIGH are the minimum and maximum
%    number of input arguments, excluding property/value pairs. VARARGIN is
%    the VARARGIN cell array provided by the client. CLIENT_NAME is the name
%    of the client.
% 
%    common_args is a structure containing arguments that are shared by ROI tools.
%    
%    specific_args is a structure containing arguments specific to a
%    particular client. These arguments include ISCLOSED for the polygon
%    tool and DRAW_API for IMRECT and IMPOINT.
    
%   Copyright 2007-2011 The MathWorks, Inc.
%   $Revision: 1.1.6.13 $  $Date: 2011/08/09 17:55:34 $
    

  common_args = struct('Position',[],...
                      'Parent',[],...
                      'Axes',[],...
                      'Fig',[],...
                      'InteractivePlacement',false,...
                      'PositionConstraintFcn',[]);
  
  specific_args = struct('DrawAPI',[],...
                         'Closed',true);
                     
                     
  
  nargin_client = length(varargin);
  
  
  % Start by parsing param/value pairs
  string_indices = find(cellfun('isclass',varargin,'char'));
  common_arg_names = {'PositionConstraintFcn'};
  valid_params = [common_arg_names,specific_arg_names];
  
  params_to_parse = ~isempty(string_indices);

  if params_to_parse
      param1_index = string_indices(1);
      num_pre_param_args = param1_index-1;
  else
      num_pre_param_args = nargin_client;
  end
  
  if num_pre_param_args > 3
      error(message('images:roiParseInputs:invalidArgumentList'))
  end

  % Enforce that number of arguments before property/value pairs is between
  % low and high number of allowed input arguments.
  checkPreParamNargin(low,high,num_pre_param_args,client_name);

  if params_to_parse

      [common_args,specific_args] = parseParamValuePairs(varargin(param1_index:end),...
          valid_params,...
          specific_arg_names,...
          num_pre_param_args,...
          client_name,...
          common_args,...
          specific_args);

  end
  
  if (num_pre_param_args == 0)
      h_parent = gca;
  else
      h_parent = varargin{1};
  end
        
  if ~ishghandle(h_parent)
      error(message('images:roiParseInputs:invalidHandle'))
  end
  
  h_axes = findAxesAncestor(h_parent,client_name);
  % At this point h_axes is guaranteed to exist, safe to use iptancestor.
  h_fig = iptancestor(h_axes,'figure');
      
  % Determine whether position was provided as two X,Y vectors or as a
  % position matrix.
  separate_xy_position = num_pre_param_args == 3;
  position_matrix = num_pre_param_args == 2;
                    
  if separate_xy_position
      x_init = varargin{2};
      y_init = varargin{3};
         
      validateVectorPosition(x_init,y_init,client_name);
               
      position = [reshape(x_init,length(x_init),1),reshape(y_init,length(y_init),1)];
      
        
  elseif position_matrix
      
      position = varargin{2};
      validateattributes(position,{'numeric'},{'2d','real'},client_name,'position',2);
      
  else
      % No position specified. Since we check earlier that the number of
      % pre-param args is not >3, this code path is exercised when 0 or 1
      % pre-param args are provided. This is for interactive placement
      % syntaxes.
      position = [];
  end
         
  % Determine whether ROI is going to be placed interactively.
  interactive_placement = isempty(position);

  common_args.Position             = position;
  common_args.Parent               = h_parent;
  common_args.Axes                 = h_axes;
  common_args.Fig                  = h_fig;
  common_args.InteractivePlacement = interactive_placement;
          
%---------------------------------------------------------   
function validateVectorPosition(x_init,y_init,client_name)
      
  validateattributes(x_init,{'numeric'},{'vector','real'},client_name,'X',1);
  validateattributes(y_init,{'numeric'},{'vector','real'},client_name,'Y',2);
  
  invalid_syntax = xor(isempty(x_init),isempty(y_init));
  if invalid_syntax
      error(message('images:roiParseInputs:invalidSyntax'))
  end
  
  unequal_vector_length = length(x_init) ~= length(y_init);
  if unequal_vector_length    
      error(message('images:roiParseInputs:invalidPosition'))
  end
         
%---------------------------------------------------------    
function h_axes = findAxesAncestor(h_parent,~)
    
  if ~ishghandle(h_parent)
      error(message('images:roiParseInputs:invalidHandle'))
  end
  
  h_axes = ancestor(h_parent, 'axes');
  if isempty(h_axes)
      error(message('images:roiParseInputs:noAxesAncestor'))
  end

%--------------------------------------------------------------  
function checkPreParamNargin(low,high,pre_param_args,~)
    
    if pre_param_args < low
        error(message('images:roiParseInputs:tooFewInputs'))
        
    elseif pre_param_args > high
        error(message('images:roiParseInputs:tooManyInputs'))
        
    end


%--------------------------------------------------------------------------  
function [common_args,specific_args] = parseParamValuePairs(in,valid_params,...
                                                  specific_arg_names,...
                                                  num_pre_param_args,...
                                                  function_name,...
                                                  common_args,...
                                                  specific_args)
        
if rem(length(in),2)~=0
    error(message('images:roiParseInputs:oddNumberArgs', upper( function_name )));
end

for k = 1:2:length(in)
    prop_string = validatestring(in{k}, valid_params, function_name,...
        'PARAM', num_pre_param_args + k);

    arg_pos = k+1+num_pre_param_args;
    
    switch prop_string
        case 'PositionConstraintFcn'
        
          validateattributes(in{k+1},{'function_handle'},{'scalar'},function_name,prop_string,arg_pos);
          common_args.(prop_string) = in{k+1};
                              
        case specific_arg_names
            % A subscript is necessary because specific_args is initialized
            % as an empty struct.
            specific_args(1).(prop_string) = in{k+1};

        otherwise
            error(message('images:roiParseInputs:unrecognizedParameter', prop_string, function_name));

    end
end
