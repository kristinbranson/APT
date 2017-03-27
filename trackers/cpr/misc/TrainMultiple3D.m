% Train the three models needed for Adam's mice in 3D.
%   - Train (if the first loaded file contains phisTr and bboxesTr)
%       + The second file must contain the frames for training (IsTr)
%       + doeq and loadH0 determine if the frames are equalized and if the
%       base histogram is loaded or computed.
%       + cpr_type: 1 for Cao et al 2013, 2 for Burgos-Artizzu et al 2013
%       (without occlusion) and 2 for Burgos-Artizzu et al 2013
%       (occlusion).
%       + regModel,regPr and prunePrm (and H0 if equalizing) are saved in
%       separate files for each trained model.
clear all
doeq = false;
loadH0 = true;
cpr_type = 2;

[file,folder]=uigetfile('.mat');
load(fullfile(folder,file));
%%
if exist('phisTr','var') && exist('bboxesTr','var')
    [fileIs,folderIs]=uigetfile('.mat');
    load(fullfile(folderIs,fileIs));
    
    % Randomly select training set
    nTr=min(numel(IsTr),20000);
    idx=randperm(numel(IsTr));
    idx=idx(1:nTr);

    if doeq
        if loadH0
            [fileH0,folderH0]=uigetfile('.mat');
            load(fullfile(folderH0,fileH0));
        else
            H=nan(256,1500);
            for i=1:1500
                H(:,i)=imhist(IsTr{idx(i)});
            end
            H0=median(H,2);
        end
        model1.H0=H0;
        for i=1:nTr,
            IsTr{idx(i)}=histeq(IsTr{idx(i)},H0);
        end    
    end
    
    % Reconstruct 3D
    imsz=size(IsTr{1});
    phisTr(:,2)=phisTr(:,2)-imsz(2)/2;
    [C,scores,latent]=pca(phisTr);
    X=phisTr*C;
    phisTr3D=X(:,1:3);
    Prm3D.C=C;
    Prm3D.X4=X(:,4);
        
    % Train first model
    [model1.regModel,model1.regPrm,model1.prunePrm]=train(phisTr(idx,:),bboxesTr(idx,:),IsTr(idx),cpr_type,'mouse_paw3D',7,25,Prm3D);
    model1.regPrm.Prm3D=Prm3D;
    model1.datafile=fullfile(folder,file);
    
    % Train model for point 1
    bboxesTr2 = [phisTr-40, 80*ones(size(phisTr2))];
    [model2.regModel,model2.regPrm,model2.prunePrm]=train(phisTr(idx,:),bboxesTr2(idx,:),IsTr(idx),cpr_type,'mouse_paw3D',7,25,Prm3D);
    model2.regPrm.Prm3D=Prm3D;
    model2.datafile=fullfile(folder,file);

    % Save models
    [file1,folder1]=uiputfile('*.mat');
    file1=fullfile(folder1,file1);
    save(file1,'-struct','model1')
    file2=[file1(1:end-4),'_pt1.mat'];
    save(file2,'-struct','model2')
else
    fprintf(1,'Wrong file');
end
