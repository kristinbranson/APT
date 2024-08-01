function MakeGTErrorPlots_addedpoints(dd,ddsavefilename,plotoutdir,MFTtbl,row4backgroundplot)


% dd - data should be landmarks x d x labeled flies
%   calculated dd =  squeeze(sqrt( sum( (Alabels - Mlabels ).^2, 2)));
% ddsavefilename - fullpath to file to save percentiles to 
% plotoutdir - output direcotry for images
% MFTtbl - table containing mov, frm, and tracked points for image of
% animal and center of fly (pA - is for 'addedpoints, would just be p for
%   normal exported label table)
% row4backgroundplot - row of the MFTtbl that will be used as reference
%   image


% rowidx for MFTtbl to use for background image
row4backgroundplot;


    prctiles = [50,75,90,95,97];
% prctiles = [10, 25, 50,75,90]
    gtprctiles = prctile(dd,prctiles,2);
    save(ddsavefilename,'gtprctiles','-append')
    [~,b] = fileparts(ddsavefilename);
    savedir = plotoutdir;
    saveimagename = fullfile(savedir,b);
    
    
moviefile = MFTtbl.mov(row4backgroundplot); 
currfrm = MFTtbl.frm(row4backgroundplot);
pts = MFTtbl.pA(row4backgroundplot,:);

h_fig = figure('Position',[427 19 600 600]);

[readframe] = get_readframe_fcn(moviefile);
flyimagefullframe = readframe(currfrm);

imagesc(flyimagefullframe)
colormap (gray)
axis equal
hold on

pts = reshape(pts,numel(pts)/2,2);
colors = [0.0         0.0        0.5       ;
       0.0         0.50392157 1.0        ;
       0.49019608, 1         0.47754586;
       1       0.58169935 0        ;
       0.5        0         0        ];
[~,numlblframes] = size(dd);
ax = gca;
% title(ax, sprintf('%d lbls, %dth, %dth, %dth, %dth, %dth',numlblframes, prctiles));
percnames2 = sprintf('%d lbls',numlblframes);
colormap (gray)
minpts = min(pts);
maxpts = max(pts);
axis([minpts(1)-35,maxpts(1)+35,minpts(2)-35,maxpts(2)+35])
box off
axis off

for j = 1:numel(prctiles)
    viscircles(pts,gtprctiles(:,j),'Color',colors(j,:),'LineWidth',1,'EnhanceVisibility',false) ;
% to use drawfly_ellipse have to put data into struct like trx

    percnames2 = sprintf('%s %dth',percnames2,prctiles(j));
    percnamesC = sprintf('%dth',prctiles(j));
    axispts = axis;
    lg = text(axispts(2)-20,axispts(3)+10+(j*7),percnamesC,'Color',colors(j,:),'FontSize',18);
    set(h_fig,'renderer','painters');
    saveas(h_fig,sprintf('%s_vec_perc%d_colorlgd.svg',saveimagename,prctiles(j)),'svg')
    saveas(h_fig,sprintf('%s_perc%d_colorlgd.jpeg',saveimagename,prctiles(j)),'jpeg')

    %delete(th);
end

end