function [hPropsPane, parameters] = newPropertiesGUI(hParent, parameters)
% Display a struct as an editable two-column property table inside hParent.
%
% Drop-in replacement for the subset of propertiesGUI() that APT actually
% uses (see ProjectSetup.advTableRefresh). The struct is recursively
% flattened into rows. Nesting depth is shown by prepending '> ' to the
% displayed property name. Leaf values are shown as strings; edits are
% parsed and validated against the leaf's original type, and reverted on
% failure.
%
% Side effect: the working copy of the struct is stored in
%   appdata(ancestor(hParent,'figure'), 'mirror')
% and updated after every successful edit. Callers (e.g. ProjectSetup)
% retrieve the latest values from there. hPropsPane is the uitable handle,
% which can be inspected, repositioned, or deleted by the caller.

  hFig = ancestor(hParent, 'figure') ;

  delete(allchild(hParent)) ;

  rows = flattenStruct_(parameters, {}, 0) ;
  tableData = rowsToTableData_(rows) ;

  pos = getpixelposition(hParent) ;
  innerWidth = max(pos(3) - 10, 50) ;
  innerHeight = max(pos(4) - 10, 50) ;
  nameColumnWidth = max(round(innerWidth * 0.55), 100) ;
  valueColumnWidth = max(innerWidth - nameColumnWidth - 30, 80) ;

  hTable = uitable(...
    'Parent', hParent, ...
    'Units', 'pixels', ...
    'Position', [5, 5, innerWidth, innerHeight], ...
    'Data', tableData, ...
    'ColumnName', {'Property', 'Value'}, ...
    'ColumnEditable', [false, true], ...
    'ColumnFormat', {'char', 'char'}, ...
    'RowName', [], ...
    'ColumnWidth', {nameColumnWidth, valueColumnWidth}, ...
    'CellEditCallback', @(src, evt) cellEdited_(src, evt, hFig)) ;
  set(hTable, 'Units', 'normalized') ;

  setappdata(hFig, 'mirror', parameters) ;
  setappdata(hFig, 'newPropsRows', rows) ;
  setappdata(hFig, 'newPropsTable', hTable) ;

  hPropsPane = hTable ;
end  % function


function cellEdited_(src, evt, hFig)
% uitable CellEditCallback: parse the user's new string against the row's
% original value type, commit on success, revert on failure.

  if isempty(evt.Indices)
    return
  end
  rowIndex = evt.Indices(1) ;
  columnIndex = evt.Indices(2) ;
  if columnIndex ~= 2
    return
  end

  rows = getappdata(hFig, 'newPropsRows') ;
  parameters = getappdata(hFig, 'mirror') ;
  rowSpec = rows(rowIndex) ;

  data = src.Data ;
  if ~rowSpec.isLeaf
    % Header rows are not editable; defensively snap back if Matlab ever
    % delivers an edit event for one anyway.
    data{rowIndex, 2} = '' ;
    src.Data = data ;
    return
  end

  oldValue = rowSpec.value ;
  newString = evt.NewData ;
  [newValue, isValid] = stringToValue_(newString, oldValue) ;

  if isValid
    parameters = setPath_(parameters, rowSpec.path, newValue) ;
    rows(rowIndex).value = newValue ;
    setappdata(hFig, 'mirror', parameters) ;
    setappdata(hFig, 'newPropsRows', rows) ;
    data{rowIndex, 2} = valueToString_(newValue) ;
  else
    data{rowIndex, 2} = valueToString_(oldValue) ;
  end
  src.Data = data ;
end  % function


function rows = flattenStruct_(s, pathSoFar, depth)
% Recursively flatten a struct into a row vector of row specs.

  rowsCell = {} ;
  rowsCell = flattenInto_(rowsCell, s, pathSoFar, depth) ;
  if isempty(rowsCell)
    rows = makeEmptyRowList_() ;
  else
    rows = vertcat(rowsCell{:}) ;
  end
end  % function


function rowsCell = flattenInto_(rowsCell, s, pathSoFar, depth)
% Recursive helper for flattenStruct_. Appends to a cell of row structs.

  if ~isstruct(s)
    return
  end
  if numel(s) > 1
    for elementIndex = 1:numel(s)
      childPath = [pathSoFar, {elementIndex}] ;
      headerName = sprintf('(%d)', elementIndex) ;
      rowsCell{end+1} = makeRow_(headerName, childPath, depth, false, []) ;  %#ok<AGROW>
      rowsCell = flattenInto_(rowsCell, s(elementIndex), childPath, depth+1) ;
    end
    return
  end
  fieldNameList = fieldnames(s) ;
  for fieldIndex = 1:numel(fieldNameList)
    fieldName = fieldNameList{fieldIndex} ;
    value = s.(fieldName) ;
    childPath = [pathSoFar, {fieldName}] ;
    if isstruct(value) && ~isempty(value)
      rowsCell{end+1} = makeRow_(fieldName, childPath, depth, false, []) ;  %#ok<AGROW>
      rowsCell = flattenInto_(rowsCell, value, childPath, depth+1) ;
    else
      rowsCell{end+1} = makeRow_(fieldName, childPath, depth, true, value) ;  %#ok<AGROW>
    end
  end
end  % function


function row = makeRow_(name, path, depth, isLeaf, value)
% Build a single row struct. Curly-brace wrapping prevents struct() from
% interpreting the cell path / cell value as array-fanout.
  row = struct('name', name, ...
               'path', {path}, ...
               'depth', depth, ...
               'isLeaf', isLeaf, ...
               'value', {value}) ;
end  % function


function rows = makeEmptyRowList_()
% Zero-length row-vector of the row-struct shape used by makeRow_().
  rows = struct('name', {}, ...
                'path', {}, ...
                'depth', {}, ...
                'isLeaf', {}, ...
                'value', {}) ;
end  % function


function tableData = rowsToTableData_(rows)
% Convert a row-spec vector into the 2-column cell array consumed by uitable.
  rowCount = numel(rows) ;
  tableData = cell(rowCount, 2) ;
  for rowIndex = 1:rowCount
    prefix = repmat('> ', 1, rows(rowIndex).depth) ;
    tableData{rowIndex, 1} = [prefix, rows(rowIndex).name] ;
    if rows(rowIndex).isLeaf
      tableData{rowIndex, 2} = valueToString_(rows(rowIndex).value) ;
    else
      tableData{rowIndex, 2} = '' ;
    end
  end
end  % function


function str = valueToString_(value)
% Convert a leaf value to its displayable string form.
  if isempty(value)
    str = '' ;
    return
  end
  if ischar(value)
    str = value ;
    return
  end
  if isstring(value)
    str = char(value) ;
    return
  end
  if islogical(value)
    if isscalar(value)
      if value
        str = 'true' ;
      else
        str = 'false' ;
      end
    else
      str = mat2str(value) ;
    end
    return
  end
  if isnumeric(value)
    if isscalar(value)
      str = num2str(value) ;
    else
      str = mat2str(value) ;
    end
    return
  end
  str = strtrim(evalc('disp(value)')) ;
end  % function


function [newValue, isValid] = stringToValue_(newString, referenceValue)
% Parse newString as a value compatible with referenceValue. A reference
% value that is empty accepts any string (the caller is responsible for any
% later type coercion -- e.g. ProjectSetup uses structLeavesStr2Double).

  if ~ischar(newString)
    newString = char(newString) ;
  end
  trimmedString = strtrim(newString) ;

  if isempty(referenceValue)
    newValue = newString ;
    isValid = true ;
    return
  end

  if islogical(referenceValue)
    switch lower(trimmedString)
      case {'true', '1', 'yes', 'on'}
        newValue = true ;
        isValid = true ;
      case {'false', '0', 'no', 'off'}
        newValue = false ;
        isValid = true ;
      otherwise
        newValue = referenceValue ;
        isValid = false ;
    end
    return
  end

  if isnumeric(referenceValue)
    parsed = str2num(trimmedString) ;  %#ok<ST2NM> need vector/matrix parsing
    if isempty(parsed) && ~isempty(trimmedString)
      newValue = referenceValue ;
      isValid = false ;
    else
      newValue = parsed ;
      isValid = true ;
    end
    return
  end

  if ischar(referenceValue) || isstring(referenceValue)
    newValue = newString ;
    isValid = true ;
    return
  end

  newValue = referenceValue ;
  isValid = false ;
end  % function


function s = setPath_(s, path, value)
% Walk the cell-array path into a (possibly-nested) struct and assign value.
% String path elements are field names; numeric ones index struct arrays.
  if isempty(path)
    s = value ;
    return
  end
  head = path{1} ;
  rest = path(2:end) ;
  if isnumeric(head)
    s(head) = setPath_(s(head), rest, value) ;
  else
    s.(head) = setPath_(s.(head), rest, value) ;
  end
end  % function
