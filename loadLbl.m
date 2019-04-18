function lbl = loadLbl(lbl_file)

tname = tempname;
mkdir(tname);
try
  fprintf('Untarring project into %s\n',tname);
  untar(lbl_file,tname);
  fprintf('... done with untar.\n');
  rawLblFile = fullfile(tname,'label_file.lbl');
catch ME
  if strcmp(ME.identifier,'MATLAB:untar:invalidTarFile')
    warningNoTrace('Label file %s is not bundled. Using it in raw (mat) format.',fname);
    rawLblFile = lbl_file;
  end
end

lbl = load(rawLblFile,'-mat');

