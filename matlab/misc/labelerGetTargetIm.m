function [im,isrotated,xdata,ydata,A,tform] = labelerGetTargetIm(labeler,mov,frm,tgt,viewi)
  isrotated = false;
  % if (int32(mov) == obj.currMovie) && (gtmode==obj.gtIsGTMode)
  %   mr = obj.movieReader(viewi) ;
  % else
  mr = MovieReader();
  mr.openForLabeler(labeler,MovieIndex(mov),viewi);
  % end
  [im,~,imRoi] = ...
    mr.readframe(frm,...
                 'doBGsub',labeler.movieViewBGsubbed,...
                 'docrop',~labeler.cropIsCropMode);
    
  % to do: figure out [~,~what to do when there are multiple views
  if ~labeler.hasTrx,
    xdata = imRoi(1:2);
    ydata = imRoi(3:4);
    A = [];
    tform = [];
  else
    ydir = get(labeler.gdata.axes_prev,'YDir');
    if strcmpi(ydir,'normal'),
      pi2sign = -1;
    else
      pi2sign = 1;
    end
  
    [x,y,th] = labeler.targetLoc(abs(mov),tgt,frm);
    if isnan(th),
      th = -pi/2;
    end
    A = [1,0,0;0,1,0;-x,-y,1]*[cos(th+pi2sign*pi/2),-sin(th+pi2sign*pi/2),0;sin(th+pi2sign*pi/2),cos(th+pi2sign*pi/2),0;0,0,1];
    tform = maketform('affine',A);  %#ok<MTFA1> 
    [im,xdata,ydata] = imtransform(im,tform,'bicubic');  %#ok<DIMTRNS> 
    isrotated = true;
  end
end  % function
  
