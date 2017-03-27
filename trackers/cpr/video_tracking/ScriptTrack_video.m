function ScriptTrack_video(i,moviefile,paramsfile)
% the parameters file contains:
%    mouse name
%    sc (scale factor)
%    model1 path
%    model2 path
%    model3 path

fid = fopen(paramsfile,'r');

mouse = fgetl(fid);
sc = str2double(fgetl(fid));
modelfile1 = fgetl(fid);
modelfile2 = fgetl(fid);
modelfile3 = fgetl(fid);

fclose(fid);

model1=load(modelfile1);
regModel1=model1.regModel; regPrm1=model1.regPrm; prunePrm1=model1.prunePrm;
if isfield(model1,'H0')
    H0=model1.H0;
end
model2=load(modelfile2);
regModel2=model2.regModel; regPrm2=model2.regPrm; prunePrm2=model2.prunePrm;
model2.file=modelfile2;
model3=load(modelfile3);
regModel3=model3.regModel; regPrm3=model3.regPrm; prunePrm3=model3.prunePrm;
model3.file=modelfile3;

modelfiles={modelfile1,modelfile2,modelfile3};
params=struct('mouse',mouse,'moviefile',moviefile,'paramsfile',paramsfile,'modelfiles',{modelfiles},'sc',sc); %#ok<NASGU>
%%
partSize=500;
% fprintf('\n**** TRACKING VIDEO %s ****\n',moviefile)
istif=strcmp(moviefile(end-2:end),'tif');
if istif
    im=imresize(imread(moviefile,1),sc);
    lmovieinfo = imfinfo(moviefile);
    nframes = numel(lmovieinfo);
    fid=0;
else
    [readframe,nframes,fid] = get_readframe_fcn(moviefile);
    if exist ('H0','var')
        im=histeq(imresize(rgb2gray_cond(readframe(1)),sc),H0);
    else
        im=imresize(rgb2gray_cond(readframe(1)),sc);
    end
end

bboxes=repmat([1 1 fliplr(size(im))],nframes,1);

% First tracking
Is=cell(partSize,1);
p=nan(nframes,4);
for t_i=1:partSize:nframes;
    t_f=min(t_i+partSize-1,nframes);
%     fprintf('\n1st tracking: frames %i-%i\n',t_i,t_f)
    for k=1:partSize
        t=t_i+k-1;
        try
            if istif
                Is{k}=imresize(imread(moviefiles_all,t),sc);
            else
                if exist('H0','var')
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

    p(t_i:t_f,:)=test_rcpr([],bboxes(t_i:t_f,:),Is,regModel1,regPrm1,prunePrm1); 
end

% Second tracking using the median of the previus tracking to
% create small bboxes.
p_med=medfilt1(p,10);
p_med(1,:)=p_med(2,:);
bboxes_med1=[p_med(:,1)-40 p_med(:,3)-40 80*ones(size(p,1),2)];
bboxes_med2=[p_med(:,2)-40 p_med(:,4)-40 80*ones(size(p,1),2)];

Is=cell(partSize,1);
p_med_bb=nan(nframes,4);
for t_i=1:partSize:nframes;
    t_f=min(t_i+partSize-1,nframes);
%     fprintf('\n2nd trackign: frames %i-%i\n',t_i,t_f)
    for k=1:partSize
        t=t_i+k-1;
        try
            if istif
                Is{k}=imresize(imread(moviefiles_all,t),sc);
            else
                if exist('H0','var')
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

    p_med_bb(t_i:t_f,[1 3])=test_rcpr([],bboxes_med1(t_i:t_f,:),Is,regModel2,regPrm2,prunePrm2); 
    p_med_bb(t_i:t_f,[2 4])=test_rcpr([],bboxes_med2(t_i:t_f,:),Is,regModel3,regPrm3,prunePrm3); 
end
d1=sqrt(sum(diff(p_med_bb(:,[1 3])).^2,2));
d2=sqrt(sum(diff(p_med_bb(:,[2 4])).^2,2));
% fprintf('\n%i jumps for point 1\n',sum(d1>20));
% fprintf('\n%i jumps for point 2\n',sum(d2>20));

if fid>0
    fclose(fid);
end
% disp('Tracking Done')
if ~exist(fullfile('/groups/branson/home/rodriguezgonzalezj/Documents/pose_tracking/mouse/Results/',mouse),'dir')
    mkdir(fullfile('/groups/branson/home/rodriguezgonzalezj/Documents/pose_tracking/mouse/Results/',mouse))
end
savefile = fullfile('/groups/branson/home/rodriguezgonzalezj/Documents/pose_tracking/mouse/Results/',mouse,['results',i,'.mat']);
save(savefile,'p_med_bb','params');

