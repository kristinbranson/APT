function create_video(p,Y,phis,dosave)
ntail=15;

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

nfids=size(p,2)/2;

fh=figure;
%set(fh,'Position',[58         168        1404         520])
set(fh,'Position',[22        1454         948         708])
him=imagesc(im);
axis image
axis off
colormap gray
hold on
hp=plot(p(1,1:nfids),p(1,nfids+1:2*nfids),'xr','MarkerSize',15,'LineWidth',2);
hpt=nan(nfids,1);
for j=1:nfids
    hpt(j)=plot(p(1,j),p(1,nfids+j),'.-r');
end
if ~isempty(Y)
    hY=plot(Y(1,1:nfids),Y(1,nfids+1:2*nfids),'xy','MarkerSize',15,'LineWidth',2);
    hYt=nan(nfids,1);
    for j=1:nfids
        hYt(j)=plot(Y(1,j),Y(1,nfids+j),'.-y');
    end
end
if ~isempty(phis)
    hph=plot(phis(1,1:nfids),phis(1,nfids+1:2*nfids),'og','MarkerSize',15,'LineWidth',2);
    hpht=nan(nfids,1);
    for j=1:nfids
        hpht(j)=plot(phis(1,j),phis(1,nfids+j),'.-g');
    end
end 

if dosave
    if ~exist(['../VID/',file(1:end-4)],'dir')
        mkdir(['../VID/',file(1:end-4)])
    end
    export_fig('-nocrop','-transparent','filename', ['../VID/',file(1:end-4),'/vimg(1).tif'])
end

for i=1:nframes
    tail=max(1,i-ntail):i;
    if strcmp(moviefile(end-2:end),'tif')
        im=imread(moviefile,(i));
    else
        im=rgb2gray_cond(readframe(i));
    end
    set(him,'CData',im)
    set(hp,'Xdata',p(i,1:nfids),'Ydata',p(i,nfids+1:2*nfids))
    for j=1:nfids
        set(hpt(j),'Xdata',p(tail,j),'Ydata',p(tail,j+nfids))
    end
    if ~isempty(Y)
        set(hY,'Xdata',Y(i,1:nfids),'Ydata',Y(i,nfids+1:2*nfids))
        for j=1:nfids
            set(hYt(j),'Xdata',Y(tail,j),'Ydata',Y(tail,j+nfids))
        end
    end
    if ~isempty(phis)
        set(hph,'Xdata',phis(i,1:nfids),'Ydata',phis(i,nfids+1:2*nfids))
        for j=1:nfids
            set(hpht(j),'Xdata',phis(tail,j),'Ydata',phis(tail,j+nfids))
        end
    end
    if dosave
        export_fig('-nocrop','-transparent','filename', ['../VID/',file(1:end-4),'/vimg(',num2str(i),'1).tif'])
    else
        drawnow
    end
end

if fid>1
    fclose(fid);
end
    
