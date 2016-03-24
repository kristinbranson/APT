%%
IDS = {
'150723_02_002_04'
'150730_02_002_01'
'150730_02_002_02'
'150730_02_002_03'
'150730_02_002_07'
'150730_02_006_02'
'150806_01_000_02'
'150828_01_002_07'
'150902_02_001_02'
'150902_02_001_07'
'151112_01_002_04'
'151112_01_002_05'
'151112_01_002_08'};

%%
diary genITrn2.txt
for id=IDS(:)',id=id{1}; %#ok<FXSET>
  [iTrn,iTstAll,iTstLbl] = td.genITrnITst2(id);
  tdIname = sprintf('tdI@13@for_%s_v2@0224.mat',id);
  save(tdIname,'iTrn','iTstAll','iTstLbl');  
end
diary off
