function lbl = loadLbl(lbl_file)

tname = tempname;
mkdir(tname);
try
  fprintf('Untarring project into %s\n',tname);
  untar(lbl_file,tname);
  fprintf('... done with untar.\n');
  rawLblFile = fullfile(tname,'label_file.lbl');
catch ME
  % Older MATLAB uses 'MATLAB:untar:invalidTarFile'; newer versions throw
  % from extractArchive with a different identifier.  Treat any untar failure
  % as "not a tar archive" and fall back to raw mat format.
  if contains(ME.identifier,'invalidTarFile') || contains(ME.message,'Invalid TAR')
    warningNoTrace('Label file %s is not bundled. Using it in raw (mat) format.',lbl_file);
    rawLblFile = lbl_file;
  else
    ME.rethrow();
  end
end

lbl = load(rawLblFile,'-mat');

[success, message, ~] = rmdir(tname,'s');
if ~success
  error('Could not clear the temp directory %s\n',message);
else
  fprintf('Cleared out temp directory %s\n',tname);
end
