% generate a video with randomly oriented squares and its corresponding trx
% file.

nframes = 30;
n = 5;
imsz = [200,200];
locs = [50 50
        100 100
        150 50
        50 150
        150 150];
sz = 10;

vid_file = '/home/mayank/temp/test_apt.avi';
trx_file = '/home/mayank/temp/test_apt_trx.trx';

%%
trx = struct();
vidObj = VideoWriter(vid_file,'Uncompressed AVI');
open(vidObj);

% f = figure();
dummy_im = zeros(imsz);
npts = size(locs,1);
for pt = 1:npts
   trx(pt).x = zeros(1,nframes);
   trx(pt).y = zeros(1,nframes);
   trx(pt).a = 5*ones(1,nframes);
   trx(pt).b = 5*ones(1,nframes);
   trx(pt).theta = zeros(1,nframes);
   trx(pt).firstframe = 1;
   trx(pt).endframe = nframes;
   trx(pt).off = 0;
end

verts = zeros(n,2,nframes,npts);
for fr = 1:nframes
%    f = figure();
   all_verts = {};
   dummy_im = zeros(imsz);
    for lndx = 1:size(locs,1)
       curx = locs(lndx,1) + randi(10);
       cury = locs(lndx,2) + randi(10);
       theta = (rand-0.5)*pi*2;
       vertices = [-sz -sz; -sz sz; sz sz;sz -sz; -sz -sz];
       R = [cos(theta) sin(theta); -sin(theta) cos(theta)];
       vertices = vertices*R;
       vertices = vertices + [curx cury];
       mask = poly2mask(vertices(:,1),vertices(:,2),imsz(1),imsz(2));
       vertices(end,:) = [curx cury];
       dummy_im(mask) = 1;
%        rect = struct;
%        rect.vertices = vertices;
%        rect.faces = [1 2 3 4];
%        patch(rect,'Vertices',rect.vertices,'FaceColor',[0 0.5 0]);
%        vertices(end+1,:) = [curx cury];
%        scatter(vertices(:,1), vertices(:,2),'+');
        trx(lndx).x(fr) = curx;
        trx(lndx).y(fr) = cury;
        trx(lndx).theta(fr) = theta;
        verts(:,:,fr,lndx) = vertices;
    end
    
%    figure(f);
%    cla;
%    imagesc(dummy_im);
%    axis('image');
%    axis('equal');
%    hold on;
%    for pt =1:npts
%       scatter(all_verts{pt}(:,1), all_verts{pt}(:,2),'+'); 
%    end
%     hold off;
%     
    
%     fr = getframe(f);
    writeVideo(vidObj,dummy_im);
end
close(vidObj);
save(trx_file,'trx','-v7.3');

%% create a lbl file with the above movie and trx file.

lbl_file = '/home/mayank/temp/test_conversion.lbl';
out_file = '/home/mayank/temp/test_conversion_updated.lbl';
L = load(lbl_file,'-mat');
L.labeledpos{1} = SparseLabelArray.create(verts,'nan');
L.labeledpostag{1} = SparseLabelArray.create(false(n,fr,npts),'log');
L.labeledposTS{1} = SparseLabelArray.create(ones(n,fr,npts)*now,'ts');
save(out_file,'-struct','L');


