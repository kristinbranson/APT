function [suspscore,tblsusp,diagstr] = outlierComputeAR_selectFile(lObj)
% FILE = '/Volumes/APT_data_flybubble/APTprojects/newlabelslist_20200510.mat';
% FILE = '/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/TouchingFlies/newlabelslist_20200510.mat';
%FILE = 'f:\arHackNavTable20171220\multitarget_bubble_expandedbehavior_20171206_cv_outlierlist.mat';
[filename,pathname] = uigetfile('*.mat', 'Select Outlier list (mov,frm,tgt)');
FILE = fullfile(pathname,filename);

fprintf('Loading file: %s\n',FILE);
load(FILE);
fprintf('Found a list of %d rows.\n',size(list,1));

list = double(list);
suspscore = cellfun(@(x)ones(size(x,3),size(x,4)),lObj.labeledpos,'uni',0);
if size(list,2) == 3
tblsusp = table(list(:,1),list(:,2),list(:,3),'VariableNames',{'mov' 'frm' 'iTgt'});
else
tblsusp = table(list(:,1),list(:,2),list(:,3),list(:,4),'VariableNames',{'mov' 'frm' 'iTgt','variable'}); 
end
diagstr = '';