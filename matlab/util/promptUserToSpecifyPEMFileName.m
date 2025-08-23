function [tfsucc, keyName, pemFileWslPath] = ...
    promptUserToSpecifyPEMFileName(keyName, pemFileWslPath)
  % Prompt user to specify pemFile
  % 
  % keyName, pemFile (in): optional defaults/best guesses
  
  if nargin<1 || isempty(keyName),
    keyName = '';
  end
  if nargin<2 || isempty(pemFileWslPath),
    pemFileWslPath = '';
  end
  
  prompt = {
    'Key name'
    'Private key (.pem or id_rsa) file WSL path'
    };
  dialogTitle = 'AWS EC2 Config';
  inputBoxWidth = 100;
  defaultValues = {keyName; pemFileWslPath} ;
  canResizeHorizontallyOnOff = 'on' ;
  browseInfo = struct('type', {'';'uigetfile'}, 'filterspec', {'';'*.pem'});
  response = ...
    inputdlgWithBrowse(prompt, ...
                       dialogTitle, ...
                       repmat([1 inputBoxWidth], [2 1]),...
                       defaultValues, ...
                       canResizeHorizontallyOnOff, ...
                       browseInfo);
  % response = ...
  %   inputdlg(prompt, ...
  %            dialogTitle, ...
  %            repmat([1 inputBoxWidth], [2 1]),...
  %            defaultValues, ...
  %            canResizeHorizontallyOnOff);
  tfsucc = ~isempty(response);      
  if tfsucc
    keyName = strtrim(response{1});
    pemFileWslPath = absolutifyFileName(strtrim(response{2}));
    if ~apt.localFileExistsAtWslPath(pemFileWslPath)
      error('Cannot find private key (.pem or id_rsa) file with WSL path %s.',pemFileWslPath);
    end
  else
    keyName = '';
    pemFileWslPath = '';
  end      
end  % function
