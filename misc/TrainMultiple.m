% Train the three models needed for Adam's mice.
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
doeq = true;
loadH0 = false;
cpr_type = 2;

addpath(genpath('/groups/branson/home/bransonk/tracking/code/piotr_toolbox_V3.02'));

defaultfolder = '/groups/branson/home/rodriguezgonzalezj/Documents/pose_tracking/mouse/Data';
defaultfile = 'mouse_M119.mat';

[file,folder]=uigetfile('.mat',sprintf('Select training file containg clicked points (e.g. %s)',defaultfile),...
  defaultfolder);
if isnumeric(file),
  return;
end
load(fullfile(folder,file),'phisTr','bboxesTr');
%%
defaultFolderIs = folder;
[~,n] = fileparts(file);
defaultFileIs = fullfile(defaultFolderIs,[n,'_Is.mat']);
if ~exist(defaultFileIs,'file'),
  defaultFileIs = defaultFolderIs;
end

[fileIs,folderIs]=uigetfile('.mat',sprintf('Select file containing images corresponding to points in %s',file),defaultFileIs);
load(fullfile(folderIs,fileIs));
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
% Train first model
[model1.regModel,model1.regPrm,model1.prunePrm]=train(phisTr(idx,:),bboxesTr(idx,:),IsTr(idx),cpr_type,'mouse_paw2',6,25);
model1.regPrm.Prm3D=[];
model1.datafile=fullfile(folder,file);

% Train model for point 1
phisTr2 = phisTr(:,[1 3]);
bboxesTr2 = [phisTr2-40, 80*ones(size(phisTr2))];
[model2.regModel,model2.regPrm,model2.prunePrm]=train(phisTr2(idx,:),bboxesTr2(idx,:),IsTr(idx),cpr_type,'mouse_paw',6,25);
model2.regPrm.Prm3D=[];
model2.datafile=fullfile(folder,file);

% Train model for point 2
phisTr3 = phisTr(:,[2 4]);
bboxesTr3 = [phisTr3-40, 80*ones(size(phisTr3))];
[model3.regModel,model3.regPrm,model3.prunePrm]=train(phisTr3(idx,:),bboxesTr3(idx,:),IsTr(idx),cpr_type,'mouse_paw',6,25);
model3.regPrm.Prm3D=[];
model3.datafile=fullfile(folder,file);

% Save models
[file1,folder1]=uiputfile('*.mat');
file1=fullfile(folder1,file1);
save(file1,'-struct','model1')
file2=[file1(1:end-4),'_pt1.mat'];
save(file2,'-struct','model2')
fil32=[file1(1:end-4),'_pt2.mat'];
save(file3,'-struct','model3')
