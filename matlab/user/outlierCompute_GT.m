function [suspscore,tblsusp,diagstr] = outlierCompute_GT(lObj)

[~,projname] = fileparts(lObj.projectfile);
GToutlierfileending = '_Optoparams20181010_gtResults_outliers5std_test.mat';
% make smarter default based on proj name? 
GToutlierfile = [lObj.projectroot,filesep,projname(1:end-18),GToutlierfileending];
if exist(GToutlierfile,'file')
    load(GToutlierfile,'tblsusp','diagstr');
    fprintf('Loaded file %s to Suspicous frames table\n',GToutlierfile)
else
    [FILE,PATH] = uigetfile([lObj.projectroot,filesep],sprintf('Select Outlier List for %s',projname));
    % [FILE,PATH] = uigetfile([lObj.projectroot,filesep],sprintf('Select Outlier List for %s',projname));
    if ~(FILE)
        suspscore = [];
        tblsusp = [];
        diagstr = [];
        return;
    end
    load(fullfile(PATH,FILE),'tblsusp','diagstr');
    fprintf('Loaded file %s to Suspicous frames table\n',FILE)
end


%populate from tblsusp? 
suspscore = cellfun(@(x)ones(size(x,3),size(x,4)),lObj.labeledpos,'uni',0);
