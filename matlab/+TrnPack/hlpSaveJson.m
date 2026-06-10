function hlpSaveJson(s,jsonoutf)
  j = jsonencode(s,'ConvertInfAndNaN',false,'PrettyPrint',true); % KB 20250524
  %jsonoutf = fullfile(packdir,jsonoutf);
  fh = fopen(jsonoutf,'w');
  fprintf(fh,'%s\n',j);
  fclose(fh);
  fprintf(1,'Wrote %s.\n',jsonoutf);
end % function
