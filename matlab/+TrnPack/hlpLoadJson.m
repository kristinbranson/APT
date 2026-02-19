function s = hlpLoadJson(jsonfile)
  jse = readtxtfile(jsonfile);
  % convert to one-line string
  jse = sprintf('%s\n',jse{:});
  s = jsondecode(jse);
  fprintf(1,'loaded %s\n',jsonfile);
end % function
