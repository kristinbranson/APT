function [regModel,regPrm,prunePrm,H0] = RCPR_simple(trainfile,imagefile,model_type)
% Train or test
%   - Train (if the first loaded file is missing IsT)
%       + The second file must contain the frames for training (IsTr)
%       + doeq and loadH0 determine if the frames are equalized and if the
%       base histogram is loaded or computed.
%       + cpr_type: 1 for Cao et al 2013, 2 for Burgos-Artizzu et al 2013
%       (without occlusion) and 2 for Burgos-Artizzu et al 2013
%       (occlusion).
%       + model_type: 'larva' (Marta's larvae with two muscles and two
%       landmarks for muscle), 'mouse_paw' (Adam's mice with one landmarks in one
%       view), 'mouse_paw2' (Adam's mice with two landmarks, one in each
%       view), 'mouse_paw3D' (Adam's mice, one landmarks in the 3D
%       reconstruction), fly_RF2 (Romain's flies, six landmarks)
%       + feature type: for 1-4 see FULL_demoRCPR.m, 5 for points in an
%       elipse with focus in any pair of landmarks, and 6 for points in a
%       circunference around each landmark.
%       + radius: dimensions of the area where features are computed, for
%       feature_type=5 is the semi-major axis relative to the distance
%       between points (recomended 1.5), for feature_type=6 is the radius
%       of the circumference (recomended 25).
%       + regModel,regPr and prunePrm (and H0 if equalizing) are the
%       variables needed for testing. 
%   - Test (if the first loaded file contains IsT)
%       + The second file must contain the frames for testing (IsT)
%       + doeq controls if the frames are equalized (it is required to load
%       a previously computed H0).
%       + The final file must contain regModel,regPr and prunePrm from a
%       previously trained model.
%       + pT conains the trackind result and lossT the loss for each frame.

% clear all
doeq = true;
loadH0 = false;
cpr_type = 2;
if nargin< 3,
model_type = 'mouse_paw';
end
feature_type = 6;
radius = 100;

load(trainfile);
%%
% Train
fprintf('Loading image file ..');
if nargin<2 || isempty(imagefile)
    [fileIs,folderIs]=uigetfile('.mat');
    load(fullfile(folderIs,fileIs));
else
  load(imagefile);
end
fprintf('Done\n');

nTr=min(numel(IsTr),20000);
idx=randperm(numel(IsTr));
idx=idx(1:nTr);

if doeq
  if loadH0
    [fileH0,folderH0]=uigetfile('.mat');
    load(fullfile(folderH0,fileH0));
  else
    H=nan(256,min(1500,numel(IsTr)));
    for i=1:size(H,2)
      H(:,i)=imhist(IsTr{idx(i)});
    end
    H0=median(H,2);
  end
  for i=1:nTr,
    IsTr{idx(i)}=histeq(IsTr{idx(i)},H0);
  end
end
[regModel,regPrm,prunePrm]=train(phisTr(idx,:),bboxesTr(idx,:),IsTr(idx),cpr_type,model_type,feature_type,radius);
regPrm.Prm3D=[];

[pTr,~,lossTr]=test_rcpr(phisTr,bboxesTr,IsTr,regModel,regPrm,prunePrm);
