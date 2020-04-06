function h = createfileinfodialog(msg,title)

h = msgbox(msg,title,'help');
fprintf('\n');
if iscell(msg),
  for i = 1:length(msg),
    fprintf('%s\n',msg{i});
  end
elseif ischar(msg),
  fprintf('%s\n',msg);
end
fprintf('\n');