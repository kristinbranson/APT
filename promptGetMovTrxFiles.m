function [tfsucc,movfile,trxfile] = promptGetMovTrxFiles(tfMultiSelect)
% [tfsucc,movfile,trxfile] = promptGetMovTrxFiles(tfMultiSelect)
%
% tfsucc: scalar logical
% movfile: cellstr
% trxfile: cellstr same size as movfile

multiSelOnOff = onIff(tfMultiSelect);

lastmov = RC.getprop('lbl_lastmovie');
[movfile,movpath] = uigetfile('*.*','Select video',lastmov,'multiselect',multiSelOnOff);
if isequal(movfile,0)
  tfsucc = false;
  movfile = [];
  trxfile = [];
  return;
end
movfile = fullfile(movpath,movfile);
movfile = cellstr(movfile);

[trxfile,trxpath] = uigetfile('*.mat','Select trx file',movpath,'multiselect',multiSelOnOff);
if isequal(trxfile,0)
  % user canceled; interpret this as "there is no trx file"
  trxfile = repmat({''},size(movfile));
else
  trxfile = fullfile(trxpath,trxfile);
  trxfile = cellstr(trxfile);
  if numel(trxfile)~=numel(movfile)
    error('promptGetMovTrxFiles:nMov',...
      'Number of trx files must match number of movie files.');
  end
end

tfsucc = true;
