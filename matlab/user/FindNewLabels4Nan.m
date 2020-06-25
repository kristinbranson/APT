%% Set parameters
% set labels mat file
fprintf('Select your exported *_labels.mat file \n \n')
[fileName,filePath] = uigetfile('*.mat','Select your exported *_labels.mat file');
labelsmatfilename = fullfile(filePath,fileName);
% labelsmatfilename = '/Users/robiea/Data/Labeler/nan_trainingprojects/multitarget_bubble_20200519_20200609_labels_TEST.mat';
% [filePath,fileName] = fileparts(labelsmatfilename);
% set date range: between date1 and date2 in format 'yyyymmdd'
date1 = '20200519';
date2 = inf;
if date2 == inf
    date2Str = num2str(date2);
else
    date2Str = date2;
end

savematfilename = fullfile(filePath,[fileName,'_LabelsFrom',date1,'to',date2Str,'.mat']);
saveoutlierlistfilename = fullfile(filePath,[fileName,'_LabelsFrom',date1,'to',date2Str,'_outlierlist','.mat']);
savemissinglabelsfilename = fullfile(filePath,[fileName,'_LabelsFrom',date1,'to',date2Str,'_missinglabels','.mat']);
% this file can be used with APT started like this: lObj = StartAPT
% run these commands and select the '_outlierlist.mat' file 
% lObj.suspSetComputeFcn(@outlierComputeAR_selectFile);
% lObj.suspComputeUI

load(labelsmatfilename);
%% find new labels
timestamps = tblLbls.pTS;
timestamps_max = max(timestamps');
dateafter = datenum(date1,'yyyymmdd');
if date2 == inf
    datebefore = inf;
else
    datebefore = datenum(date2,'yyyymmdd');
end
[a,idx] = find(timestamps_max >= dateafter & timestamps_max <= datebefore);
newlabels = tblLbls(idx,:);


%% output 'outlier' file list of new labels
movielist = unique(tblLbls.mov,'stable');
movienum = 1:numel(movielist);

newtouchinglabels_list =[];
for i = 1:height(newlabels)
    newtouchinglabels_list(i,1) = movienum(strcmp(newlabels.mov{i},movielist));
    newtouchinglabels_list(i,2) = newlabels.frm(i);
    newtouchinglabels_list(i,3) = newlabels.iTgt(i);
    newtouchinglabels_list(i,4) = max(newlabels.pTS(i));
end
% list = newtouchinglabels_list;

[~,b] = sortrows(newtouchinglabels_list,[1,2]);
list = newtouchinglabels_list(b,:);
% list(:,1:3)
save(saveoutlierlistfilename,'list');

%% find touching labels, and close flies without both labeled
% mov, frm, target

newlabelsmat = list(:,1:3);
[movie_unique,IA,IC] = unique(newlabelsmat(:,1),'stable');
movie_count = histc(newlabelsmat(:,1),movie_unique);

 missinglabels =[];
 touchinglabels = 0;
% loop over all frames in each movie
for i = 1:numel(movie_unique)
    idx = [];
    idx = find(IC == i);
    % load trxfile for movie i
    moviename = movielist{movie_unique(i)};
    expdir = fileparts(moviename);
    trxfile = fullfile(expdir,'registered_trx.mat');
    load(trxfile,'trx')
    % for each row in movie 1, find 'touching' flies
    for j = 1:numel(idx)
        flyj = newlabelsmat(idx(j),3);
        frmj = newlabelsmat(idx(j),2);
        [flydist,flyID] = computeflydistance4allflies(flyj,frmj,trx);
        fidx = [];
        fidx = find(flydist <= 4.2);
        closeflies = flyID(fidx);
        if closeflies > 0
            touchinglabels = touchinglabels + 1; 
        end 
        for k = 1:numel(closeflies)
           labeledpartner = find(newlabelsmat(:,1) == movie_unique(i) & newlabelsmat(:,2) == frmj & newlabelsmat(:,3) == closeflies(k));
           if isempty(labeledpartner)
               missinglabels = [ missinglabels; movie_unique(i),frmj,flyj,closeflies(k)];
           end
        end 
        
    end   
end
list = missinglabels;
save(savemissinglabelsfilename,'list');



%% print stats to command line


fprintf('%s has: \n',fileName)

% number of new labels

fprintf('%d total new labels \n',rows(newlabels))

% new labels per movie
for i = 1:numel(movie_unique)
  fprintf('movie %d has %d new labels \n',movie_unique(i),movie_count(i))
  
end

% number of 'touching fly' labels per movie

fprintf('%d labeled flies have a nearby fly \n',touchinglabels)
fprintf('%d labeled flies have a nearby unlabeled fly \n',size(missinglabels,1))
% output just new labels mat file
save(savematfilename,'newlabels')
% output just new labels csv file

% movie name, frame, target

