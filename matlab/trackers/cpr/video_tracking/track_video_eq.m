function [p,Y,lossT,lossY]=track_video_eq(domerge,dotest,bboxes,sc)
if nargin<3
    bboxes=[];
    sc=1;
end
filetypes={  '*.ufmf','MicroFlyMovieFormat (*.ufmf)'; ...
      '*.fmf','FlyMovieFormat (*.fmf)'; ...
      '*.sbfmf','StaticBackgroundFMF (*.sbfmf)'; ...
      '*.avi','AVI (*.avi)'
      '*.mp4','MP4 (*.mp4)'
      '*.mov','MOV (*.mov)'
      '*.mmf','MMF (*.mmf)'
      '*.tif','TIF (*.tif)'
      '*.*','*.*'};
[file,folder]=uigetfile(filetypes);
moviefile=fullfile(folder,file);
if strcmp(moviefile(end-2:end),'tif')
    im=imresize(imread(moviefile,1),sc);
    lmovieinfo = imfinfo(moviefile);
    nframes = numel(lmovieinfo);
    fid=0;
else
    [readframe,nframes,fid] = get_readframe_fcn(moviefile);
    im=imresize(rgb2gray_cond(readframe(1)),sc);
end
[modelfile,modelfolder]=uigetfile('.mat');
load(fullfile(modelfolder,modelfile),'regModel','regPrm','prunePrm');

partSize=300;
if isempty(bboxes)
    bboxes=repmat([1 1 fliplr(size(im))],nframes,1);
elseif size(bboxes,1)==1
    bboxes=repmat(bboxes,nframes,1);
end
if dotest
    [filetest,foldertest]=uigetfile(filetypes);
    load(fullfile(foldertest,filetest));
end

[fileeq,foldereq]=uigetfile(filetypes);
    load(fullfile(foldertest,filetest));

Is=cell(partSize,1);
p=nan(nframes,2*regModel.model.nfids);
pR=nan(nframes,2*regModel.model.nfids,5);
for t_i=1:partSize:nframes;
    t_f=min(t_i+partSize-1,nframes);
    fprintf('\nTRACKING FRAMES %i-%i\n',t_i,t_f)
    for i=1:partSize
        t=t_i+i-1;
        try
            if strcmp(moviefile(end-2:end),'tif')
                Is{i}=imresize(imread(moviefile,t),sc);
            else
                Is{i}=imresize(rgb2gray_cond(readframe(t)),sc);
            end
        catch
            Is(i:end)=[];
            break
        end
    end
    [p(t_i:t_f,:),pR(t_i:t_f,:,:)]=test_rcpr([],bboxes(t_i:t_f,:),Is,regModel,regPrm,prunePrm); 
end

if domerge
    good=ones(size(pR,1),1);
    bboxes(:,5)=100;
    bboxes=num2cell(bboxes,2);
    Y=performTracking(pR,bboxes,good,-1);
else
    Y=[];
end

if dotest && exist(phisT,'var') && ~isempty(phisT)
   isT=all(~isnan(phisT),2);
   phisT=phisT(isT);
   pT=p(isT);
   lossT = shapeGt('dist',regModel.model,pT,phisT);
   if ~isempty(Y)
       YT=Y(isT);
       lossY = shapeGt('dist',regModel.model,YT,phisT);
   else
       lossY=[];
   end
else
    lossT=[];
    lossY=[];
end

if fid>1
    fclose(fid);
end

