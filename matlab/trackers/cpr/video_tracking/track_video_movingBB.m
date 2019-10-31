function [p,lossT,Y,lossY]=track_video_movingBB(domerge,dotest,bbox1)
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
    lmovieinfo = imfinfo(moviefile);
    nframes = numel(lmovieinfo);
    fid=0;
else
    [readframe,nframes,fid] = get_readframe_fcn(moviefile);
end
[modelfile,modelfolder]=uigetfile('.mat');
load(fullfile(modelfolder,modelfile),'regModel','regPrm','prunePrm');

if dotest
    [filetest,foldertest]=uigetfile(filetypes);
    load(fullfile(foldertest,filetest));
end

nfids=regModel.model.nfids;
p=nan(nframes,2*nfids);
pR=nan(nframes,2*regModel.model.nfids,5);
bboxes=nan(nframes,4);
bboxes(1,:)=bbox1;
tic
for t=1:nframes;
    if strcmp(moviefile(end-2:end),'tif')
        Is{1}=imread(moviefile,t);
    else
        Is{1}=rgb2gray_cond(readframe(t));
    end
    [p(t,:),pR(t,:,:)]=test_rcpr([],bboxes(t,:),Is,regModel,regPrm,prunePrm); 
    bbcx=mean(p(t,1:nfids)); bbcy=mean(p(t,nfids+1:end));
    bboxes(t+1,:)=[bbcx-bbox1(3)/2 bbcy-bbox1(4)/2 bbox1(3:4)];
    if rem(t,100)==0
        fprintf('Frame # %i.\nElapsed time: %f\n\n',t,toc)
    end
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

