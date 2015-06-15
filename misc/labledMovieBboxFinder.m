[file,folder]=uigetfile;
file = fullfile(folder,file);
lmovieinfo = imfinfo(file);
clear framebbox
framebbox = cell(size(2, numel(lmovieinfo)));
labeledmovie = zeros(lmovieinfo(1).Height,lmovieinfo(1).Width, numel(lmovieinfo));
se = strel('disk',10);
Is=cell(numel(lmovieinfo),1);
for i = 1:numel(lmovieinfo)
    tic
    ltempimage = imread(file,'Index',i,'Info',lmovieinfo);
    Is{i}=ltempimage;
    thresh = min(prctile(ltempimage(:),94));
    ltempimage = ltempimage>thresh;
    ltempimage = imclose(ltempimage, se);
    ltempimage = imfill(ltempimage,'holes');
    cc = bwconncomp(ltempimage);
    ltempimage2 = zeros(size(ltempimage));
    cellsizes = (cellfun(@numel,(cc.PixelIdxList)));
    maxcellindex = [];
    [maxcell,maxcellindex] = max(cellsizes);
    ltempimage2(cc.PixelIdxList{maxcellindex}) = ltempimage(cc.PixelIdxList{maxcellindex});
    bbox = regionprops(ltempimage2,'BoundingBox');
    bbox = struct2cell(bbox);
    bbox = cell2mat(bbox);
    framebbox{1,i} = i;
    framebbox{2,i} = bbox;
    toc
%     imagesc(ltempimage2); 
%     axis image
%     hold on; 
%     plot([bbox(1) bbox(1)+bbox(3)], [bbox(2) bbox(2)],'m-');
%     plot([bbox(1) bbox(1)], [bbox(2) bbox(2)+bbox(4)],'m-');
%     plot([bbox(1) bbox(1)+bbox(3)], [bbox(2)+bbox(4) bbox(2)+bbox(4)],'m-');
%     plot([bbox(1)+bbox(3) bbox(1)+bbox(3)], [bbox(2) bbox(2)+bbox(4)],'m-');
% 
%     labeledmovie(:,:,i) = ltempimage2;
%     drawnow;
end

%save('labeledmoviebbox', 'framebbox')