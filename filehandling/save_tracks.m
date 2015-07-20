% succeeded = save_tracks(trx,matname,...)
% optional arguments:
% 'doappend' = false
% 'varname' = trx
function succeeded = save_tracks(trx,matname,varargin)

[doappend,varname,timestamps] = myparse(varargin,'doappend',false,'varname','trx','timestamps',[]);

succeeded = false;

if isfield(trx,'f2i'),
  trx = rmfield(trx,'f2i');
end

try
  if ~strcmp(varname,'trx'),
    eval(sprintf('%s = trx;',varname));
  end
  if doappend && exist(matname,'file'),
    save('-append',matname,varname,'timestamps');
  else
    save(matname,varname,'timestamps');
  end
catch ME
  s=sprintf('Error saving %s to %s. Error info:\n',varname,matname);
  write_log(1,getappdata(0,'experiment',s)
  disp(ME);
  return;
end
succeeded = true;