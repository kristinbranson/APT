function CleanupSplitTracking(lblFile,infoFile,outTrkFile,outLogFile)

if ~exist(outTrkFile,'file'),
  error('Output track file %s does not exist.',outTrkFile);
end

[info,isdone] = CheckSplitTracking(lblFile,infoFile);
if ~all(isdone),
  error('Not all jobs are done.\n');
end

fid = fopen(outLogFile,'w');
for i = 1:numel(info),
  fprintf(fid,'*****\n%s\n*****\n',info(i).logfile);
  fidin = fopen(info(i).logfile,'r');
  while true,
    s = fgetl(fidin);
    if ~ischar(s),
      break;
    end
    fprintf(fid,[s,'\n']);
  end
  fclose(fidin);
  fprintf(fid,'\n\n\n');
end
fclose(fid);
assert(exist(outLogFile,'file')>0);

for i = 1:numel(info),
  delete(info(i).logfile);
  delete(info(i).shfile);
  delete(info(i).trkFile);
end
  