% Track multiple videos in 3D. 
% There are three steps:
%   1. Main tracking: All the landmarks are tracked simultaneously (even if
%   there are multiple views)
%   2. The results are smoothed using a median filter.
%   3. A bounding cube of side 80 centered in each of the points form step 2.
%   Each point is the tracked individually using the the bounding cubes.
% 
% It requires the following input files:
%   - Text file containing the path to each experiment file (if there are
%   labels to compare the result) or experiment directory (if there are no
%   labels)
%   - Model trained to track all the landmarks simultaneusly (must contain
%   regModel, regPrm and prunePrm)
%   - One model trained per landmark to track each landmark individually using a
%   bounding box of side 80 (must contain regModel, regPrm and prunePrm).
% Input parameters
%   - dotest: compare results with manually labeled frames
%   - doequ: euqlize images using a reference histogram. The the histogram
%   H0 must be stored in Model1.
%   - sc: scale factor (1 to use the original resolution
%   - partSize: number of frames tracked in each step
%  Output:
%   - p_all: cell arrain containing tracking results (one cell per video).
%   - moviefiles_all: video locations
%   - params: some setup information
function track_video_bb3D_all(dotest,doeq,sc,partSize)
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

params=struct('expfile',expfile,'modelfiles',{modelfile1,modelfile2},'sc',sc); 

if isfield(regPrm1.Prm3D,'bboxes0');
    bboxes0=regPrm1.Prm3D.bboxes0;
    regPrm1.Prm3D=rmfield(regPrm1.Prm3D,'bboxes0');
else
    bboxes0= [-24.9672904302395,137.158942682351,-179.630494327754,320.141953401281,192.140976406931,152.979390532612];
end


partSize=300;
%%
fid = fopen(expfile,'r');
% [expdirs_all,moviefiles_all,labeledpos]=read_exp_list_labeled(fid,dotest);
[expdirs_all,moviefiles_all,labeledpos,dotest]=read_exp_list_NONlabeled(fid);

nfiles=numel(moviefiles_all);
p3D=cell(nfiles,1);
p_med3D=cell(nfiles,1);
p_med_bb3D=cell(nfiles,1);
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
    p3D{i}=nan(nframes,3);
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

        p3D{i}(t_i:t_f,:)=test_rcpr3D(bboxes(t_i:t_f,:),Is,regModel1,regPrm1,prunePrm1); 
    end
        
        % Second tracking using the median of the previus tracking to
    % create small bboxes.
    p_med3D{i}=medfilt1(p3D{i},10);
    p_med3D{i}(1,:)=p_med3D{i}(2,:);
    bboxes_med=[p_med3D{i}-40  80*ones(size(p3D{i}))];

    Is=cell(partSize,1);
    p_med_bb3D{i}=nan(nframes,3);
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

        p_med_bb3D{i}(t_i:t_f,:)=test_rcpr3D(bboxes_med(t_i:t_f,:),Is,regModel2,regPrm2,prunePrm2); 
    end
    p_all{i}=[p_med_bb3D{i} mean(regPrm1.Prm3D.X4)*ones(size(p_med_bb3D{i},1),1)]/regPrm1.Prm3D.C;
    p_all{i}(:,2)=p_all{i}(:,2)+imsz(2)/2;
    p_all{i}=round(p_all{i});

    
    d1=sqrt(sum(diff(p_all{i}(:,[1 3])).^2,2));
    d2=sqrt(sum(diff(p_all{i}(:,[2 4])).^2,2));
    fprintf('\n%i jumps for point 1\n',sum(d1>20));
    fprintf('\n%i jumps for point 2\n',sum(d2>20));
    if dotest
        modeltest = shapeGt('createModel','mouse_paw2');
        labeledpos{i}=reshape(labeledpos{i},4,[])';
        islabeled=~any(isnan(labeledpos{i}),2);
        labeledidx=find(islabeled);
        phis{i,1}=labeledpos{i}(islabeled,:);
        phis{i,2}=p_all{i}(islabeled,:);
        loss{i}=shapeGt('dist',modeltest,phis{i,1},phis{i,2});
        fprintf('\n%i frames with loss > 10\n',sum(loss{i}>10))
    end
    
        
    if fid>0
        fclose(fid);
    end
end

[Sfile,Sfolder] = uiputfile('*.mat');
save(fullfile(Sfolder,Sfile),'p_all','moviefiles_all','params');

disp('Tracking Done')

