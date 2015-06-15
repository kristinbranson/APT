function ScriptTrack_video_all(params)
% the parameters file contains:
%    mouse name
%    dotest (true/false)
%    sc (scale factor)
%    expfile path
%    model1 path
%    model2 path
%    model3 path
fid = fopen(params,'r');

mouse = fgetl(fid);
dotest = str2num(fgetl(fid));
sc = str2double(fgetl(fid));
expfile = fgetl(fid);
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
model3=load(modelfile3);
regModel3=model3.regModel; regPrm3=model3.regPrm; prunePrm3=model3.prunePrm;



partSize=500;
%%
fid = fopen(expfile,'r');
%[expdirs_all,moviefiles_all,labeledpos]=read_exp_list_labeled(fid,dotest);
[~,moviefiles_all,labeledpos,dotest]=read_exp_list_NONlabeled(fid);

nfiles=numel(moviefiles_all);
p=cell(nfiles,1);
p_med=cell(nfiles,1);
p_med_bb=cell(nfiles,1);
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
        if exist ('H0','var')
            im=histeq(imresize(rgb2gray_cond(readframe(1)),sc),H0);
        else
            im=imresize(rgb2gray_cond(readframe(1)),sc);
        end
    end
    
    bboxes=repmat([1 1 fliplr(size(im))],nframes,1);
        
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

        p{i}(t_i:t_f,:)=test_rcpr([],bboxes(t_i:t_f,:),Is,regModel1,regPrm1,prunePrm1); 
    end
        
    % Second tracking using the median of the previus tracking to
    % create small bboxes.
    p_med{i}=medfilt1(p{i},10);
    p_med{i}(1,:)=p_med{i}(2,:);
    bboxes_med1=[p_med{i}(:,1)-40 p_med{i}(:,3)-40 80*ones(size(p{i},1),2)];
    bboxes_med2=[p_med{i}(:,2)-40 p_med{i}(:,4)-40 80*ones(size(p{i},1),2)];

    Is=cell(partSize,1);
    p_med_bb{i}=nan(nframes,4);
    for t_i=1:partSize:nframes;
        t_f=min(t_i+partSize-1,nframes);
        fprintf('\n2nd trackign: frames %i-%i\n',t_i,t_f)
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

        p_med_bb{i}(t_i:t_f,[1 3])=test_rcpr([],bboxes_med1(t_i:t_f,:),Is,regModel2,regPrm2,prunePrm2); 
        p_med_bb{i}(t_i:t_f,[2 4])=test_rcpr([],bboxes_med2(t_i:t_f,:),Is,regModel3,regPrm3,prunePrm3); 
    end
    d1=sqrt(sum(diff(p_med_bb{i}(:,[1 3])).^2,2));
    d2=sqrt(sum(diff(p_med_bb{i}(:,[2 4])).^2,2));
    fprintf('\n%i jumps for point 1\n',sum(d1>20));
    fprintf('\n%i jumps for point 2\n',sum(d2>20));
    if dotest
        labeledpos{i}=reshape(labeledpos{i},4,[])';
        islabeled=~any(isnan(labeledpos{i}),2);
        phis{i,1}=labeledpos{i}(islabeled,:);
        phis{i,2}=p_med_bb{i}(islabeled,:);
        loss{i}=shapeGt('dist',regModel2.model,phis{i,1},phis{i,2});
        fprintf('\n%i frames with loss > 10\n',sum(loss{i}>10))
    end
        
    if fid>0
        fclose(fid);
    end
end
disp('Tracking Done')

p_all = p_med_bb;

savefile = fullfile('/groups/branson/home/rodriguezgonzalezj/Documents/pose_tracking/mouse/Results/',[mouse,datestr(now,'_',TimestampFormat),'.mat']);
save(savefile,'p_all','moviefiles_all');

