clear 
[file,folder]=uigetfile('.mat');
load(fullfile(folder,file));
[modelfile,modelfolder]=uigetfile('.mat');
load(fullfile(modelfolder,modelfile),'regModel','regPrm','prunePrm');
[pT,pRT,lossT]=test_rcpr(phisT,bboxesT,IsT,regModel,regPrm,prunePrm); 

good=ones(size(pRT,1),1);
bboxesT(:,5)=100;
bboxesT=num2cell(bboxesT,2);
Y=performTracking(pRT,bboxesT,good,-1);

