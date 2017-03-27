% Function for tracking a single video. The video is tracked partSize
% frames at a time 
% - domerge: merge the results as in faceTracking.m (Burgos-Artizzu et al.,
% 2103)
% - dotest: compare the results with manually tracked labels and compute
% the loss.
% - bboxes: bonding boxes
%      + empty to use the whole image
%      + [x, y, width, height] to use the same box in every frame
%      + nFrames x 4 matrix to use a different box for each frame
% - sc: scaling factor, 1 to use the original resolution.

function [p,Y,lossT,lossY,bad,pR]=track_video(varargin)

[moviefile,model,H0_file,do_eq,domerge,dotest,bboxes,sc] = myparse(varargin,...
  'moviefile','',...
  'model','',...
  'H0_file','',...
  'do_eq',true,...
  'domerge',false,...
  'dotest',false,...
  'bboxes',[],...
  'sc',1);
partSize=300;

if nargin<3
    bboxes=[];
    sc=1;
end
if isempty(moviefile)
filetypes={  '*.ufmf','MicroFlyMovieFormat (*.ufmf)'; ...
      '*.fmf','FlyMovieFormat (*.fmf)'; ...
      '*.sbfmf','StaticBackgroundFMF (*.sbfmf)'; ...
      '*.avi','AVI (*.avi)'
      '*.mp4','MP4 (*.mp4)'
      '*.mov','MOV (*.mov)'
      '*.mmf','MMF (*.mmf)'
      '*.tif','TIF (*.tif)'
      '*.*','*.*'};
  [file,folder]=uigetfile(filetypes,'movie file');
moviefile=fullfile(folder,file);
end
if strcmp(moviefile(end-2:end),'tif')
    im=imresize(imread(moviefile,1),sc);
    lmovieinfo = imfinfo(moviefile);
    nframes = numel(lmovieinfo);
    fid=0;
else
    [readframe,nframes,fid] = get_readframe_fcn(moviefile);
    im=imresize(rgb2gray_cond(readframe(1)),sc);
end
if isempty(model)
  [modelfile,modelfolder]=uigetfile('.mat','regression model');
  load(fullfile(modelfolder,modelfile));
else
  load(model);
end

if do_eq && ~exist('H0','var')
  if isempty(H0_file)
    [fileH0,folderH0]=uigetfile('.mat','H0 (equalization) file');
    load(fullfile(folderH0,fileH0));
    
  else
    load(H0_file);
  end
end

if isempty(bboxes)
    bboxes=repmat([1 1 fliplr(size(im))],nframes,1);
elseif size(bboxes,1)==1
    bboxes=repmat(bboxes,nframes,1);
end
if dotest
    [filetest,foldertest]=uigetfile(filetypes);
    load(fullfile(foldertest,filetest));
end

Is=cell(partSize,1);
p=nan(nframes,regModel.model.D*regModel.model.nfids);
pR=nan(nframes,regModel.model.D*regModel.model.nfids,prunePrm.numInit);

bad = nan(nframes,1);
for t_i=1:partSize:nframes;
    t_f=min(t_i+partSize-1,nframes);
    fprintf('\nTRACKING FRAMES %i-%i\n',t_i,t_f)
    for i=1:partSize
        t=t_i+i-1;
        try
            if strcmp(moviefile(end-2:end),'tif')
                Is{i}=imresize(imread(moviefile,t),sc);
            else
              if do_eq
                Is{i}=histeq(imresize(rgb2gray_cond(readframe(t)),sc),H0);
              else
                Is{i}=imresize(rgb2gray_cond(readframe(t)),sc);
              end                
            end
        catch
            Is(i:end)=[];
            break
        end
    end
    [p(t_i:t_f,:),pR(t_i:t_f,:,:),~,fail]=test_rcpr([],bboxes(t_i:t_f,:),Is,regModel,regPrm,prunePrm); 
    if ~isempty(fail)
      bad(fail + t_i - 1) = 1;
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

