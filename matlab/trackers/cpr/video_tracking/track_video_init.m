function [p,lossT]=track_video_init(dotest,bboxes,p1)
if nargin<3
    bboxes=[];
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
    im=imread(moviefile,1);
    lmovieinfo = imfinfo(moviefile);
    nframes = numel(lmovieinfo);
    fid=0;
else
    [readframe,nframes,fid] = get_readframe_fcn(moviefile);
    im=rgb2gray_cond(readframe(1));
end
[modelfile,modelfolder]=uigetfile('.mat');
load(fullfile(modelfolder,modelfile),'regModel','regPrm','prunePrm','H0');

if isempty(bboxes)
    bboxes=repmat([1 1 fliplr(size(im))],nframes,1);
elseif size(bboxes,1)==1
    bboxes=repmat(bboxes,nframes,1);
end
if dotest
    [filetest,foldertest]=uigetfile(filetypes);
    load(fullfile(foldertest,filetest));
end

p=nan(nframes,3*regModel.model.nfids);
p(1,:)=p1;
tic
for t=2:nframes;
    if strcmp(moviefile(end-2:end),'tif')
        Is{1}=imread(moviefile,t);
    else
        Is{1}=histeq(rgb2gray_cond(readframe(t)),H0);
    end
    pi=repmat(p(t-1,:),[1,1,5])+5*randn([size(p(t-1,:)),5]); %temp
    p(t,:)=test_rcpr([],bboxes(t,:),Is,regModel,regPrm,prunePrm,pi); 
    if rem(t,100)==0
        fprintf('Frame # %i.\nElapsed time: %f\n\n',t,toc)
    end
end

if dotest && exist(phisT,'var') && ~isempty(phisT)
   isT=all(~isnan(phisT),2);
   phisT=phisT(isT);
   pT=p(isT);
   lossT = shapeGt('dist',regModel.model,pT,phisT);
else
    lossT=[];
end

if fid>1
    fclose(fid);
end

