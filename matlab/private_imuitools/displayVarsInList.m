function displayVarsInList(ws_vars, hListBox, disp_fields)
%displayVarsInList Displays the workspace variable structure in a list box. 
%  displayVarsInList(HLISTBOX, WS_VARS) displays the name, size and class of the
%  variables listed in the WS_VARS structure into a listbox with handle,
%  HLISTBOX.
%  
%  displayVarsInList(HLISTBOX, WS_VARS,'name') displays the only name of the
%  variables listed in WS_VARS into the listbox, HLISTBOX. 
  
  
%  ('name' is simply just a placeholder)
  
%   Copyright 2004 The MathWorks, Inc.  
%   $Revision: 1.1.8.1 $  $Date: 2004/08/10 01:50:04 $
  
    
  ws_vars = orderfields(ws_vars);
  num_of_vars = length(ws_vars);
  
  display_name_only = nargin == 3;


  var_str = cell(num_of_vars,1);
  var_names = {ws_vars.name};
  longest_var_name = max(cellfun('length',var_names));
  format1 = sprintf('%%-%ds',longest_var_name+2);
  format2 = sprintf('%%-12s %%-6s');
  format_all = sprintf('%s%s',format1,format2);
  
  
  if display_name_only
    for n = 1:num_of_vars
      var_str{n} = sprintf(' %s\n',ws_vars(n).name);
    end
    
  else
    for n = 1:num_of_vars
      if length(ws_vars(n).size) == 3
        sz_str = sprintf('%dx%dx%d',ws_vars(n).size);
        tmp_str= sprintf(format_all,ws_vars(n).name,...
                         sz_str,ws_vars(n).class);
      else
        sz_str = sprintf('%dx%d',ws_vars(n).size);
        tmp_str= sprintf(format_all,ws_vars(n).name,...
                         sz_str,ws_vars(n).class);
      end
      %for k = 1:num_of_fields
      %  tmp_str = sprintf('%s %-10s',tmp_str, ws_vars(n).(disp_fields{k}));
      %end
      var_str{n} = sprintf('%s\n',tmp_str);
    end
  end
 set(hListBox,'String',var_str);
 set(hListBox,'HorizontalAlignment','left');
 
end %displayVarsInList

