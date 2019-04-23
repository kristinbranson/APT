function markerParams=cookMarkerParams(markerParamsRaw)
% Take an unprocessed markerParamsRaw structure, and 'cook' it to make sure
% it conforms to the proper format.

% Process nextra_markers field
if isfield(markerParamsRaw,'nextra_markers') && ~isempty(markerParamsRaw.nextra_markers),
  nextra_markers=markerParamsRaw.nextra_markers;
else
  nextra_markers=1;
end
    
% Process extra_markersize field
extra_markersize = repmat(12,[1,nextra_markers]);
if isfield(markerParamsRaw,'extra_markersize') && ~isempty(markerParamsRaw.extra_markersize),
  raw_extra_markersize=markerParamsRaw.extra_markersize;
  % convert to cell array, if needed
  if isnumeric(raw_extra_markersize) ,
    raw_extra_markersize_as_numbers=raw_extra_markersize;
  else
    if ischar(raw_extra_markersize),
      raw_extra_markersize_as_cell_array_of_strings = regexp(raw_extra_markersize,',','split');
    elseif iscell(raw_extra_markersize)
      raw_extra_markersize_as_cell_array_of_strings = raw_extra_markersize;
    end
    raw_extra_markersize_as_numbers= ...
      cellfun(@str2double,raw_extra_markersize_as_cell_array_of_strings);
  end
  % process each el of the cell array, converting to double and storing in
  % extra_markersize if conversion works
  nRaw=length(raw_extra_markersize_as_numbers);
  if nextra_markers<nRaw
    raw_extra_markersize_as_numbers=raw_extra_markersize_as_numbers(1:nextra_markers);
  end
  elementIsFinite=isfinite(raw_extra_markersize_as_numbers);
  extra_markersize(elementIsFinite)=raw_extra_markersize_as_numbers(elementIsFinite);
end

% Process extra_marker field
extra_marker = repmat({'o'},[1,nextra_markers]);
if isfield(markerParamsRaw,'extra_marker') && ~isempty(markerParamsRaw.extra_marker),
  raw_extra_marker=markerParamsRaw.extra_marker;
  if ischar(raw_extra_marker),
    raw_extra_marker_as_cell_array = regexp(raw_extra_marker,',','split');
  else
    raw_extra_marker_as_cell_array = raw_extra_marker;
  end
  nRaw=numel(raw_extra_marker_as_cell_array);
  if nRaw < nextra_markers,
    % warndlg('Number of extra marker entries less than number of extra markers', ...
    %         'Problem parsing trxGraphicParams.extra_marker');
    extra_marker = [raw_extra_marker_as_cell_array repmat({'None'},[1,nextra_markers-nRaw])];
  else
    extra_marker = raw_extra_marker_as_cell_array(1:nextra_markers);
  end
end

% Process extra_linestyle field
extra_linestyle = repmat({'-'},[1,nextra_markers]);
if isfield(markerParamsRaw,'extra_linestyle') && ~isempty(markerParamsRaw.extra_linestyle),
  raw_extra_linestyle=markerParamsRaw.extra_linestyle;
  if ischar(raw_extra_linestyle),
    extra_linestyle = regexp(raw_extra_linestyle,',','split');
  else
    extra_linestyle = raw_extra_linestyle;
  end
  nPresently=numel(extra_linestyle);
  if nPresently < nextra_markers,
    % warndlg('Number of extra linestyle entries less than number of extra markers', ...
    %         'Problem parsing trxGraphicParams.extra_linestyle');
    extra_linestyle = ...
      [extra_linestyle repmat({'None'},[1,nextra_markers-nPresently])];
  else
    extra_linestyle = extra_linestyle(1:nextra_markers);
  end  
end

% Package everything up
markerParams=struct('nextra_markers',{nextra_markers}, ...
                    'extra_markersize',{extra_markersize}, ...
                    'extra_marker',{extra_marker}, ...
                    'extra_linestyle',{extra_linestyle});

end
