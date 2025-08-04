lbldir = '/groups/branson/bransonlab/aniket/APT/label_files';
outdir = 'tmpdata';
lblfiles = {'walshjTest_July_25.lbl','Nmanakov52_July_18.lbl'};
outmatfile = fullfile(outdir,'combined_table.mat');

inlblfile_combined = '/groups/branson/bransonlab/aniket/APT/label_files/walshjTest_July_25.lbl';
outlblfile_combined = fullfile(outdir,'combined.lbl');

lObj = StartAPT;
VARNAME = 'tblLbls';
tbldata = cell(size(lblfiles));
for i = 1:numel(lblfiles),
  inlblfile = fullfile(lbldir,lblfiles{i});
  lObj.projLoadGUI(inlblfile);
  tbldata{i} = lObj.labelGetMFTableLabeled('useMovNames',true);
end

tbldata_combined = tbldata{1};
for i = 2:numel(lblfiles),
  ism = tblismember(tbldata{i},tbldata_combined,MFTable.FLDSID);
  tbldata_combined = tblvertcatsafe(tbldata_combined,tbldata{i}(~ism,:));
end

%% for python manipulation
% output combined label file if you need to do some python stuff with it

% convert to struct
s_combined = labelTable2struct(tbldata_combined);
% output to hdf5 file -- can mess around with in python
save(outmatfile,s_combined);

% do some stuff in python
% ...

s_combined = load(outmatfile);
tbldata_combined = struct2table(s_combined);

%% load into the combined lbl file

lObj.projLoadGUI(inlblfile_combined);
lObj.labelPosBulkImportTbl(tbldata_combined);
lObj.projSave(outlblfile_combined);