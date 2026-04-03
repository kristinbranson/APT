function print_struct_code(s, var_name)
  % Print MATLAB code to recreate a scalar struct
  % 
  % Usage:
  %   print_struct_code(s)           % uses variable name 's'
  %   print_struct_code(s, 'myvar')  % uses variable name 'myvar'
  %
  % Arguments:
  %   s        - scalar struct to recreate
  %   var_name - string name for the root struct variable (optional, default 's')
  %
  % Handles:
  %   - Scalar structs (recursive)
  %   - 1-d double arrays
  %   - 1-d cell arrays
  %   - 1-d char arrays (strings)
  %   - LabelMode objects
  %   - 1-d logical arrays
  
  if nargin < 2
    var_name = 's';
  end
  
  if ~isstruct(s) || ~isscalar(s)
    error('Input must be a scalar struct');
  end
  
  % Always start with struct() declaration
  fprintf('%s = struct();\n', var_name);
  
  field_names = fieldnames(s);
  
  if isempty(field_names)
    return;
  end
  
  for i = 1:length(field_names)
    field_name = field_names{i};
    field_value = s.(field_name);
    field_ref = sprintf('%s.%s', var_name, field_name);
    
    print_value_assignment(field_ref, field_value);
  end
end

function print_value_assignment(var_ref, value)
  % Print assignment for a single value
  
  if isstruct(value)
    if ~isscalar(value)
      error('Only scalar structs are supported');
    end
    % Recursive case for nested struct
    field_names = fieldnames(value);
    if isempty(field_names)
      fprintf('%s = struct();\n', var_ref);
    else
      for i = 1:length(field_names)
        field_name = field_names{i};
        field_value = value.(field_name);
        nested_ref = sprintf('%s.%s', var_ref, field_name);
        print_value_assignment(nested_ref, field_value);
      end
    end
    
  elseif isnumeric(value)
    if ~isvector(value) && ~isempty(value)
      error('Only 1-d double arrays are supported');
    end
    if isempty(value)
      fprintf('%s = [];\n', var_ref);
    elseif isscalar(value)
      fprintf('%s = %.17g;\n', var_ref, value);
    else
      % 1-d array - preserve row/column orientation
      if size(value, 1) == 1
        % Row vector
        value_str = sprintf('%.17g ', value);
        value_str = value_str(1:end-1); % remove trailing space
        fprintf('%s = [%s];\n', var_ref, value_str);
      else
        % Column vector
        value_str = sprintf('%.17g; ', value);
        value_str = value_str(1:end-2); % remove trailing '; '
        fprintf('%s = [%s];\n', var_ref, value_str);
      end
    end
    
  elseif islogical(value)
    if ~isvector(value) && ~isempty(value)
      error('Only 1-d logical arrays are supported');
    end
    if isempty(value)
      fprintf('%s = logical([]);\n', var_ref);
    elseif isscalar(value)
      if value
        fprintf('%s = true;\n', var_ref);
      else
        fprintf('%s = false;\n', var_ref);
      end
    else
      % 1-d logical array - preserve row/column orientation
      logical_strs = cell(size(value));
      for i = 1:length(value)
        if value(i)
          logical_strs{i} = 'true';
        else
          logical_strs{i} = 'false';
        end
      end
      if size(value, 1) == 1
        % Row vector
        value_str = strjoin(logical_strs, ', ');
        fprintf('%s = [%s];\n', var_ref, value_str);
      else
        % Column vector
        value_str = strjoin(logical_strs, '; ');
        fprintf('%s = [%s];\n', var_ref, value_str);
      end
    end
    
  elseif iscell(value)
    if ~isvector(value) && ~isempty(value)
      error('Only 1-d cell arrays are supported');
    end
    if isempty(value)
      fprintf('%s = {};\n', var_ref);
    else
      % Preserve row/column orientation for cell arrays
      if size(value, 1) == 1
        % Row cell array
        fprintf('%s = {', var_ref);
        for i = 1:length(value)
          if i > 1
            fprintf(', ');
          end
          print_cell_element(value{i});
        end
        fprintf('};\n');
      else
        % Column cell array
        fprintf('%s = {', var_ref);
        for i = 1:length(value)
          if i > 1
            fprintf('; ');
          end
          print_cell_element(value{i});
        end
        fprintf('};\n');
      end
    end
    
  elseif ischar(value)
    if ~isvector(value) && ~isempty(value)
      error('Only 1-d char arrays are supported');
    end
    % Escape single quotes by doubling them
    escaped_value = strrep(value, '''', '''''');
    fprintf('%s = ''%s'';\n', var_ref, escaped_value);
    
  elseif isa(value, 'LabelMode')
    if ~isscalar(value)
      error('Only scalar LabelMode objects are supported');
    end
    % Get the enumeration name by finding which enumeration value matches
    enum_values = enumeration('LabelMode');
    enum_names = {'NONE', 'SEQUENTIAL', 'TEMPLATE', 'HIGHTHROUGHPUT', ...
                  'MULTIVIEWCALIBRATED2', 'MULTIANIMAL', 'SEQUENTIALADD'};
    
    match_idx = [];
    for i = 1:length(enum_values)
      if value == enum_values(i)
        match_idx = i;
        break;
      end
    end
    
    if isempty(match_idx)
      error('Unknown LabelMode value');
    end
    
    fprintf('%s = LabelMode.%s;\n', var_ref, enum_names{match_idx});
    
  else
    error('Unsupported data type: %s', class(value));
  end
end

function print_cell_element(element)
  % Print a single cell array element
  
  if isnumeric(element)
    if isempty(element)
      fprintf('[]');
    elseif isscalar(element)
      fprintf('%.17g', element);
    else
      if ~isvector(element)
        error('Only 1-d double arrays are supported');
      end
      % Preserve row/column orientation
      if size(element, 1) == 1
        % Row vector
        value_str = sprintf('%.17g ', element);
        value_str = value_str(1:end-1); % remove trailing space
        fprintf('[%s]', value_str);
      else
        % Column vector
        value_str = sprintf('%.17g; ', element);
        value_str = value_str(1:end-2); % remove trailing '; '
        fprintf('[%s]', value_str);
      end
    end
    
  elseif islogical(element)
    if isempty(element)
      fprintf('logical([])');
    elseif isscalar(element)
      if element
        fprintf('true');
      else
        fprintf('false');
      end
    else
      if ~isvector(element)
        error('Only 1-d logical arrays are supported');
      end
      logical_strs = cell(size(element));
      for i = 1:length(element)
        if element(i)
          logical_strs{i} = 'true';
        else
          logical_strs{i} = 'false';
        end
      end
      % Preserve row/column orientation
      if size(element, 1) == 1
        % Row vector
        value_str = strjoin(logical_strs, ', ');
        fprintf('[%s]', value_str);
      else
        % Column vector
        value_str = strjoin(logical_strs, '; ');
        fprintf('[%s]', value_str);
      end
    end
    
  elseif ischar(element)
    if ~isvector(element) && ~isempty(element)
      error('Only 1-d char arrays are supported');
    end
    % Escape single quotes by doubling them
    escaped_element = strrep(element, '''', '''''');
    fprintf('''%s''', escaped_element);
    
  elseif isa(element, 'LabelMode')
    if ~isscalar(element)
      error('Only scalar LabelMode objects are supported');
    end
    % Get the enumeration name by finding which enumeration value matches
    enum_values = enumeration('LabelMode');
    enum_names = {'NONE', 'SEQUENTIAL', 'TEMPLATE', 'HIGHTHROUGHPUT', ...
                  'MULTIVIEWCALIBRATED2', 'MULTIANIMAL', 'SEQUENTIALADD'};
    
    match_idx = [];
    for i = 1:length(enum_values)
      if element == enum_values(i)
        match_idx = i;
        break;
      end
    end
    
    if isempty(match_idx)
      error('Unknown LabelMode value');
    end
    
    fprintf('LabelMode.%s', enum_names{match_idx});
    
  elseif isstruct(element)
    error('Nested structs in cell arrays are not supported');
    
  elseif iscell(element)
    error('Nested cell arrays are not supported');
    
  else
    error('Unsupported data type in cell array: %s', class(element));
  end
end