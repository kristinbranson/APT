% Train using a 3D reconstruction
%  - Loads labels (phisTr) and images (Istr), does the 3D reconstruction
%  (PCA of 2*nVies dimensions to 3 dimensions) and trains.
%  - if dobb=true the bounding box will be a cube of size 80 centered
%  around the label, else the bounding box will be big enough to contain
%  all the labels.
%  - doeq and loadH0 determine if the frames are equalized and if the
%  base histogram is loaded or computed.
%  - cpr_type: 1 for Cao et al 2013, 2 for Burgos-Artizzu et al 2013
%  (without occlusion) and 2 for Burgos-Artizzu et al 2013
%  (occlusion).
%  + model_type: 'mouse_paw3D' (Adam's mice, one landmarks in the 3D
%  reconstruction).
%  + feature type: 7 for points in a circunference around each landmark,
%  reprojected in 2D (randomly chooses one of the views).
%  + radius: radius of the circunference.
%  + regModel,regPr and prunePrm (and H0 if equalizing) are the
%  variables needed for testing. 
clear

dobb=0; %train with bbozes centered in the label
doeq = false;
loadH0=0; 
cpr_type = 2;
model_type = 'mouse_paw3D';
feature_type = 7;
radius = 25;


% labels
[file,folder]=uigetfile('.mat');
load(fullfile(folder,file));

% images
[fileIs,folderIs]=uigetfile('.mat');
load(fullfile(folderIs,fileIs));

% Reconstruct 3D
imsz=size(IsTr{1});
phisTr(:,2)=phisTr(:,2)-imsz(2)/2;
[C,scores,latent]=pca(phisTr);
X=phisTr*C;
phisTr3D=X(:,1:3);
Prm3D.C=C;
Prm3D.X4=X(:,4);

% bboxes
if dobb
    bboxesTr=[phisTr3D-40 80*ones(size(phisTr3D))];
else
    bboxes0=[min(phisTr3D)-10 max(phisTr3D)-min(phisTr3D)+20];
%     bboxes0= [-24.9672904302395,137.158942682351,-179.630494327754,320.141953401281,192.140976406931,152.979390532612];
    bboxesTr=repmat(bboxes0,numel(IsTr),1);
end

% Randomly select training set
nTr=min(numel(IsTr),20000);
idx=randperm(numel(IsTr));
idx=idx(1:nTr);
% idx=5740:7559;

% equalize
if doeq
    if loadH0
        [fileH0,folderH0]=uigetfile('.mat');
        load(fullfile(folderH0,fileH0));
    else
        H=nan(256,1500);
        nsamples=1500;
        framessample=linspace(1,numel(idx),nsamples);
        for i=1:nsamples
            H(:,i)=imhist(IsTr{idx(i)});
        end
        H0=median(H,2);
    end
end

for i=1:numel(idx),
    IsTr{idx(i)}=histeq(IsTr{idx(i)},H0);
end    


% train
[regModel,regPrm,prunePrm]=train(phisTr3D(idx,:),bboxesTr(idx,:),IsTr(idx,:),2,'mouse_paw3D',7,25,Prm3D);
if exist('bboxes0','var')
    regPrm.Prm3D.bboxes0=bboxes0;
end

pTr3D=test_rcpr3D(bboxesTr(idx,:),IsTr(idx,:),regModel,regPrm,prunePrm);
pTr=[pTr3D mean(regPrm.Prm3D.X4)*ones(size(pTr3D,1),1)]/regPrm.Prm3D.C;
modeltest = shapeGt('createModel','mouse_paw2');
loss=shapeGt('dist',modeltest,pTr,phisTr(idx,:));