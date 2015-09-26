function [tfsucc,movfile,trxfile] = promptGetMovTrxFiles
% [tfsucc,movfile,trxfile] = promptGetMovTrxFiles

lastmov = RC.getprop('lbl_lastmovie');
[movfile,movpath] = uigetfile('*.*','Select video',lastmov);
if ~ischar(movfile)
  tfsucc = false;
  movfile = [];
  trxfile = [];
  return;
end
movfile = fullfile(movpath,movfile);

[trxfile,trxpath] = uigetfile('*.mat','Select trx file',movpath);
if ~ischar(trxfile)
  % user canceled; interpret this as "there is no trx file"
  trxfile = [];
else
  trxfile = fullfile(trxpath,trxfile);
end

tfsucc = true;
