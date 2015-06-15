% Track multiple videos in 3D (main tracking) and 2D (secondary tracking). 
% There are three steps:
%   1. Main tracking: All the landmarks are tracked simultaneously (even if
%   there are multiple views)
%   2. The results are reprojected in 2D and smoothed using a median filter.
%   3. A bounding  of side 80 centered in each of the points form step 2.
%   Each point is the tracked individually usint the the bounding boxes.
% 
% It requires the following input files:
%   - Text file containing the path to each experiment file (if there are
%   labels to compare the result) or experiment directory (if there are no
%   labels)
%   - Model trained to track all the landmarks simultaneusly in 3D (must contain
%   regModel, regPrm and prunePrm)
%   - One model trained per landmark to track each landmark individually using a
%   bounding box of side 80 (must contain regModel, regPrm and prunePrm).
% Input parameters
%   - dotest: compare results with manually labeled frames
%   - doequ: euqlize images using a reference histogram. The the histogram
%   H0 must be stored in Model1.
%   - sc: scale factor (1 to use the original resolution
%   - partSize: number of frames tracked in each step
%   - bbox0: bounding box size (recomended for Adam's mice
%   [-25, 137.158942682351,-180,320,192,153])
%  Output:
%   - p_all: cell arrain containing tracking results (one cell per video).
%   - moviefiles_all: video locations
%   - params: some setup information
function track_video_3D_all(dotest,doeq,sc,partSize,bboxes0)

[file,folder]=uigetfile('*.txt');
expfile=fullfile(folder,file);

[modelfile1,modelfolder1]=uigetfile('.mat');
modelfile1=fullfile(modelfolder1,modelfile1);
model1=load(modelfile1);
regModel1=model1.regModel; regPrm1=model1.regPrm; prunePrm1=model1.prunePrm;
if isfield(model1,'H0')
    H0=model1.H0;
end
[modelfile2,modelfolder2]=uigetfile('.mat');
modelfile2=fullfile(modelfolder2,modelfile2);
model2=load(modelfile2);
regModel2=model2.regModel; regPrm2=model2.regPrm; prunePrm2=model2.prunePrm;
[modelfile3,modelfolder3]=uigetfile('.mat');
modelfile3=fullfile(modelfolder3,modelfile3);
model3=load(modelfile3);
regModel3=model3.regModel; regPrm3=model3.regPrm; prunePrm3=model3.prunePrm;

params=struct('expfile',expfile,'modelfiles','modelfiles',{modelfile1,modelfile2,modelfile3},'sc',sc); 


partSize=300;
%%
fid = fopen(expfile,'r');
if dotest
    [expdirs_all,moviefiles_all,labeledpos]=read_exp_list_labeled(fid,dotest);
else
    [expdirs_all,moviefiles_all,labeledpos,dotest]=read_exp_list_NONlabeled(fid);
end

nfiles=numel(moviefiles_all);
p=cell(nfiles,1);
p_med=cell(nfiles,1);
p_all=cell(nfiles,1);
phis=cell(nfiles,2);
loss=cell(nfiles,1);

%%
for i=1:nfiles
    fprintf('\n**** TRACKING VIDEO %s ****\n',moviefiles_all{i})
    istif=strcmp(moviefiles_all{i}(end-2:end),'tif');
    if istif
        im=imresize(imread(moviefiles_all,1),sc);
        lmovieinfo = imfinfo(moviefiles_all{i});
        nframes = numel(lmovieinfo);
        fid=0;
    else
        [readframe,nframes,fid] = get_readframe_fcn(moviefiles_all{i});
        im=histeq(imresize(rgb2gray_cond(readframe(1)),sc),H0);
    end
    imsz=size(im);
    
    bboxes=repmat(bboxes0,nframes,1);
        
    % First tracking
    Is=cell(partSize,1);
    p{i}=nan(nframes,4);
    for t_i=1:partSize:nframes;
        t_f=min(t_i+partSize-1,nframes);
        fprintf('\n1st tracking: frames %i-%i\n',t_i,t_f)
        for k=1:partSize
            t=t_i+k-1;
            try
                if istif
                    Is{k}=imresize(imread(moviefiles_all,t),sc);
                else
                    if doeq && exist('H0','var')
                        Is{k}=histeq(imresize(rgb2gray_cond(readframe(t)),sc),H0);
                    else
                        Is{k}=imresize(rgb2gray_cond(readframe(t)),sc);
                    end
                end
            catch
                Is(k:end)=[];
                break
            end
        end

        p3D(t_i:t_f,:)=test_rcpr3D(bboxes(t_i:t_f,:),Is,regModel1,regPrm1,prunePrm1); 
    end
    p{i}=[p3D mean(regPrm1.Prm3D.X4)*ones(size(p3D,1),1)]/regPrm1.Prm3D.C;
    p{i}(:,2)=p{i}(:,2)+imsz(2)/2;
        
        % Second tracking using the median of the previus tracking to
    % create small bboxes.
    p_med{i}=medfilt1(p{i},10);
    p_med{i}(1,:)=p_med{i}(2,:);
    bboxes_med1=[p_med{i}(:,1)-40 p_med{i}(:,3)-40 80*ones(size(p{i},1),2)];
    bboxes_med2=[p_med{i}(:,2)-40 p_med{i}(:,4)-40 80*ones(size(p{i},1),2)];

    Is=cell(partSize,1);
    p_all{i}=nan(nframes,4);
    for t_i=1:partSize:nframes;
        t_f=min(t_i+partSize-1,nframes);
        fprintf('\n2nd trackign: frames %i-%i\n',t_i,t_f)
        for k=1:partSize
            t=t_i+k-1;
            try
                if istif
                    Is{k}=imresize(imread(moviefiles_all,t),sc);
                else
                    if doeq && exist('H0','var')
                        Is{k}=histeq(imresize(rgb2gray_cond(readframe(t)),sc),H0);
                    else
                        Is{k}=imresize(rgb2gray_cond(readframe(t)),sc);
                    end

                end
            catch
                Is(k:end)=[];
                break
            end
        end

        p_all{i}(t_i:t_f,[1 3])=test_rcpr([],bboxes_med1(t_i:t_f,:),Is,regModel2,regPrm2,prunePrm2); 
        p_all{i}(t_i:t_f,[2 4])=test_rcpr([],bboxes_med2(t_i:t_f,:),Is,regModel3,regPrm2,prunePrm3); 
    end
    p_all{i}=round(p_all{i});
    
    d1=sqrt(sum(diff(p_all{i}(:,[1 3])).^2,2));
    d2=sqrt(sum(diff(p_all{i}(:,[2 4])).^2,2));
    fprintf('\n%i jumps for point 1\n',sum(d1>20));
    fprintf('\n%i jumps for point 2\n',sum(d2>20));
    if dotest
        labeledpos{i}=reshape(labeledpos{i},4,[])';
        islabeled=~any(isnan(labeledpos{i}),2);
        labeledidx=find(islabeled);
        phis{i,1}=labeledpos{i}(islabeled,:);
        phis{i,2}=p_all{i}(islabeled,:);
        loss{i}=shapeGt('dist',regModel2.model,phis{i,1},phis{i,2});
        fprintf('\n%i frames with loss > 10\n',sum(loss{i}>10))
    end
    
        
    if fid>0
        fclose(fid);
    end
end
[Sfile,Sfolder] = uiputfile('*.mat');
save(fullfile(Sfolder,Sfile),'p_all','moviefiles_all','params');

disp('Tracking Done')

