clear
nT=100;
ntrack = 5;

[file,folder] = uigetfile('*.mat');
File=fullfile(folder,file);


load(File,'phis','Is','bboxes');
bboxes=round(bboxes);
%%
phisNaN = find(any(isnan(phis),2));
phis(phisNaN,:) = [];
Is(phisNaN) = [];
bboxes(phisNaN,:) = [];


pTr = cell(ntrack,1);
lossTr = cell(ntrack,1);
pT = cell(ntrack,1);
lossT = cell(ntrack,1);
regModel = cell(ntrack,1);
regPrm = cell(ntrack,1);
prunePrm = cell(ntrack,1);
idxrand=randperm(numel(Is)); 
idx=[1:150,304:403];
idx=idx(randperm(length(idx)));
idx=[idx,177:278];
for i = 1:ntrack
    
    nTr = i*50;
    % Traning data
    idxTr=idx(1:nTr);
    phisTr=phis(idxTr,:);
    bboxesTr=bboxes(idxTr,:);
    IsTr=Is(idxTr);
    

    % Test data
    idxT=idx(end-nT+1:end);
    phisT=phis(idxT,:);
    bboxesT=bboxes(idxT,:);
    IsT=Is(idxT);
    
    [pTr{i},lossTr{i},pT{i},lossT{i},regModel{i},regPrm{i},prunePrm{i}]=larva_muscles(phisTr,bboxesTr,IsTr,phisT,bboxesT,IsT);
end

meanlossTr = cellfun(@mean,lossTr);
meanlossT = cellfun(@mean,lossT);

figure
plot((1:ntrack)*50,meanlossTr,(1:ntrack)*50,meanlossT,'r')

medlossTr = cellfun(@median,lossTr);
medlossT = cellfun(@median,lossT);

figure
plot((1:ntrack)*50,medlossTr,(1:ntrack)*50,medlossT,'r')

