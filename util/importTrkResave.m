function importTrkResave(lObj,lblname)
%Script to update .trk tracking info in an APT project.  Use when you
%have re-run tracking but want to keep project structure (backup project files first)
%
%Run like this:
% lObj = Labeler;
% importTrkResave(lObj,'/full/path/to/lbl/file')

%set to 1 if want to put date at end of filename and keep old file (annoying but safe)
%set to 0 if just want to overwrite old file (not annoying but dangerous)
putDateAtEndOfFilename=0;

lObj.projLoad(lblname);
fprintf(1,'Loaded proj: %s\n',lblname);
lObj.labels2ImportTrkPromptAuto();

if putDateAtEndOfFilename ==1
    time = clock;
    [p,f,e] = fileparts(lblname);
    lblname2 = fullfile(p,[f, '_trk_' ,num2str(time(1)),num2str(time(2)),num2str(time(3)), e]);
else
    lblname2 = lblname;
end

lObj.projSaveRaw(lblname2);
fprintf(1,'Saved proj: %s\n',lblname2);

