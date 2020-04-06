function compare3dMovie(outf,fmov,smov,matfilef,matfiles,outfile)
%%

  [readframe_front,~] = get_readframe_fcn(fmov);
  [readframe_side,~] = get_readframe_fcn(smov);
  rdf = load(matfilef);
  rds = load(matfiles);
  FRONTOUTPUTTYPE = 1; % output of MRF
  SIDEOUTPUTTYPE = 1; % output of raw detector
  rdf.locs = permute(rdf.locs(:,:,FRONTOUTPUTTYPE,:),[1,2,4,3]);
  rds.locs = permute(rds.locs(:,:,SIDEOUTPUTTYPE,:),[1,2,4,3]);
  r3d = load(outfile);
  rdf3.locs = r3d.pfrontbest_re;
  rds3.locs = r3d.psidebest_re;
  
%%
  fig = figure(3);
  set(fig,'Position',[400,500,1500,600])
  vidobj = VideoWriter(outf);
  open(vidobj);

  hax = createsubplots(1,3,0.025);
  cc = jet(5);
  colors = cc;
  Pbest = r3d.Pbest;
  
  for ndx = 1:size(rdf3.locs,3),
    fim = readframe_front(ndx);
    sim = readframe_side(ndx);
    hold(hax(1),'off');
    hold(hax(2),'off');
    him_front = imshow(fim,'Parent',hax(1));
    him_side = imshow(sim,'Parent',hax(2));
    hold(hax(1),'on')
    hold(hax(2),'on')
    axis(hax(1),'image','off');
    axis(hax(2),'image','off');
    
    scatter(rdf3.locs(1,:,ndx),rdf3.locs(2,:,ndx),50,cc,'x','Parent',hax(1));
    for i = 1:5
      plot([rdf3.locs(1,i,ndx),rdf.locs(ndx,i,1)],[rdf3.locs(2,i,ndx),rdf.locs(ndx,i,2)],'c-','Parent',hax(1));
    end
    scatter(rds3.locs(1,:,ndx),rds3.locs(2,:,ndx),50,cc,'x','Parent',hax(2));
    for i = 1:5
      plot([rds3.locs(1,i,ndx),rds.locs(ndx,i,1)],[rds3.locs(2,i,ndx),rds.locs(ndx,i,2)],'c-','Parent',hax(2));
    end

    
    nlandmarks = 5;
    h3 = nan(1,nlandmarks);
    hold(hax(3),'off');
    for i = 1:nlandmarks,
      h3(i) = plot3(hax(1,3),squeeze(Pbest(1,i,1:ndx)),squeeze(Pbest(2,i,1:ndx)),squeeze(Pbest(3,i,1:ndx)),'-',...
        'Color',colors(i,:),'LineWidth',1);
      if i == 1,
        hold(hax(3),'on');
      end
    end
    minP = min(min(Pbest,[],2),[],3);
    maxP = max(max(Pbest,[],2),[],3);
    dP = maxP-minP;
    axis(hax(1,3),'equal');
    set(hax(1,3),'XLim',[minP(1)-dP(1)/20,maxP(1)+dP(1)/20],...
      'YLim',[minP(2)-dP(2)/20,maxP(2)+dP(2)/20],...
      'ZLim',[minP(3)-dP(3)/20,maxP(3)+dP(3)/20]);
    xlabel(hax(1,3),'X');
    ylabel(hax(1,3),'Y');
    zlabel(hax(1,3),'Z');
    grid(hax(1,3),'on');

      

    drawnow;
    fr = getframe(fig);
    writeVideo(vidobj,fr);
    if mod(ndx,100)==0,fprintf('%d/%d\n',ndx,size(rdf3.locs,3)); end

  end
close(vidobj);
