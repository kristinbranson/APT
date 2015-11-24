classdef RCPR
  methods (Static)
    
    function vizPts(ax,im,pts)
      % pts: npts x d
      cla(ax,'reset');

      npts = size(pts,1);
      colors = jet(npts);

      imagesc(im,'Parent',ax,[0,255]);
      axis(ax,'image','off');
      hold(ax,'on');
      colormap gray;
      for iPt = 1:npts
        plot(ax,pts(iPt,1),pts(iPt,2),'wo','MarkerFaceColor',colors(iPt,:));
      end
    end
    
    function hAx = vizPtsTile(Is,pts,nr,nc)
      % pts: either npts x d x ntrl, OR ntrl x (npts*d)
      
      nTrl = numel(Is);
      
      switch ndims(pts)
        case 2
          pts = permute(pts,[2 1]);
          d = 2;
          npts = size(pts,1)/d;
          assert(npts==round(npts));
          pts = reshape(pts,[npts d nTrl]);
        case 3
          [~,~,nTrlPts] = size(pts);
          assert(nTrlPts==nTrl);
        otherwise
          assert(false);
      end
          
      
      figure(7);
      clf;
      nplot = nr*nc;
      idxPlt = randsample(nTrl,nplot);
      hAx = createsubplots(nr,nc,.01);
      for iPlt = 1:numel(idxPlt)
        iTrl = idxPlt(iPlt);
        ax = hAx(iPlt);
        RCPR.vizPts(ax,Is{iTrl},pts(:,:,iTrl));

        [imnr,imnc] = size(Is{iTrl});
        text(imnc-10,imnr-10,num2str(iTrl),'parent',ax);
      end
    end
    
    function vizPts2(ax,im,pts1,pts2)
      cla(ax,'reset');
      
      if isrow(pts1)
        pts1 = reshape(pts1,numel(pts1)/2,2);
      end
      if isrow(pts2)
        pts2 = reshape(pts2,numel(pts2)/2,2);
      end

      imagesc(im,'Parent',ax,[0,255]);
      axis(ax,'image','off');
      hold(ax,'on');
      colormap gray;

      COLORS = {[0 0 1] [0 1 0]};
      npts1 = size(pts1,1);
      for iPt = 1:npts1
        plot(ax,pts1(iPt,1),pts1(iPt,2),'o','MarkerFaceColor',COLORS{1});
      end
      npts2 = size(pts2,1);
      for iPt = 1:npts2
        plot(ax,pts2(iPt,1),pts2(iPt,2),'o','MarkerFaceColor',COLORS{2});
      end
    end
    
  end
end