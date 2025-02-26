function [tfsucc,keyName,pemFile] = ...
    promptUserToSpecifyPEMFileName(keyName, pemFile)
  % Prompt user to specify pemFile
  % 
  % keyName, pemFile (in): optional defaults/best guesses
  
  if nargin<1 || isempty(keyName),
    keyName = '';
  end
  if nargin<2 || isempty(pemFile),
    pemFile = '';
  end
  
  PROMPT = {
    'Key name'
    'Private key (.pem or id_rsa) file'
    };
  NAME = 'AWS EC2 Config';
  INPUTBOXWIDTH = 100;
  BROWSEINFO = struct('type',{'';'uigetfile'},'filterspec',{'';'*.pem'});

  resp = inputdlgWithBrowse(PROMPT,NAME,repmat([1 INPUTBOXWIDTH],2,1),...
    {keyName;pemFile},'on',BROWSEINFO);
  tfsucc = ~isempty(resp);      
  if tfsucc
    keyName = strtrim(resp{1});
    pemFile = strtrim(resp{2});
    if exist(pemFile,'file')==0
      error('Cannot find private key (.pem or id_rsa) file %s.',pemFile);
    end
  else
    keyName = '';
    pemFile = '';
  end      
end  % function

