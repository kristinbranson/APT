classdef CropInfo < handle
  
  % Note on 'roi', 'posn', and cropping
  %
  % roi
  % This refers to an APT-style roi. This is [xlo xhi ylo yhi] with integer 
  % values. In normal situations, we will want positive integer values, 
  % which are also <= the image size. We call these rois 'proper rois'. 
  %
  % We imagine pixel centers being located at integer coordinates. Pixels
  % on the boundary of the roi are included in the roi. The ROI
  % [30 40 50 80] has 11 columns and 31 rows. 
  %
  % posn
  % This is a MATLAB/imcrop- or imrect-style rectangle position. This is
  % [xlo ylo width height] with floating pt values. 
  % (xlo,ylo,xlo+width,ylo+width are the coordinate positions of the 
  % infinitely thin edges of a rectangle. 
  %
  % IMPORTANT: Note that a posn with width w will convert to an roi with 
  % width w+1. Converting back to a posn will return to the width w.
  %
  % posnCtrd
  % This is an imrect-style rectangle specification like posn; 
  % [xctr yctr width height]. 
  %
  % widthHeight
  % This refers to the imrect-style width/height as in a posn or posnCtrd.
  % IMPORTANT: the width/height for a corresponding roi IS ONE LARGER in
  % each dim.
  %
  % When MATLAB crops using eg imcrop, the resulting rectangle does not 
  % need to have width cols and height rows, because any pixel even 
  % slightly enclosed by the rectangle is included in the crop.
  %
  % APT crops differently. When a posn [xlo ylo width height] is used to
  % crop, the resulting ROI is guaranteed to have round(width)+1 cols and 
  % round(height)+1 rows.
  
  properties
    roi % [1x4] 
  end
  methods 
    function obj = CropInfo(roiArr)
      % roi: [nobj x 4]
      
      if nargin==0
        obj.roi = nan(1,4);
        return;
      end
      
      assert(size(roiArr,2)==4);
      n = size(roiArr,1);
      if n==1
        obj.roi = roiArr;        
      else
        for i=n:-1:1
          obj(i,1) = CropInfo(roiArr(i,:));
        end
      end
    end
  end
  methods (Static)
    function obj = CropInfoCentered(posnCtrd)
      % Create array of CropInfos given posncntred. See notes at top of
      % CropInfo for definitions.
      %
      % posnCtrd: [nx4]. Each row is [xctr yctr width height], can be
      % non-integers.
      
      assert(size(posnCtrd,2)==4);
      posn = CropInfo.rectPosCtrd2Pos(posnCtrd);
      roi = CropInfo.rectPos2roi(posn);     
      obj = CropInfo(roi);
    end
  end
  methods (Static)
    function tf = roiIsProper(roi,imnc,imnr)
      %
      %
      % roi: [nx4]
      % imnc, imnr: either [nx1], or scalars
      %
      % tf: [nx1] logical
      
      tf = 1<=roi(:,1) & roi(:,2)<=imnc & ...
           1<=roi(:,3) & roi(:,4)<=imnr;
    end
    function [roi,tfchanged] = roiSmartFit(roi,imnc,imnr)
      % Adjust an roi so it is proper, ie it fits within the image
      % [1,imnc] x [1,imnr].
      %
      % An error is thrown if the roi cannot fit.
      %
      % roi (in): [1x4]
      % imnc: scalar
      % imnr: scalar
      % 
      % roi (out): [1x4]
      % tfchanged: true if roi has been modified
      
      roinc = roi(2)-roi(1)+1;
      roinr = roi(4)-roi(3)+1;
      if roinc>imnc || roinr>imnr
        error('ROI cannot fit in image.');
      end
      
      tfchanged = false;
      if roi(1)<1
        roi([1 2]) = roi([1 2]) + 1 - roi(1);
        tfchanged = true;
      end
      if roi(2)>imnc
        roi([1 2]) = roi([1 2]) - roi(2) + imnc;
        tfchanged = true;
      end
      if roi(3)<1
        roi([3 4]) = roi([3 4]) + 1 - roi(3);
        tfchanged = true;
      end
      if roi(4)>imnr
        roi([3 4]) = roi([3 4]) - roi(4) + imnr;
        tfchanged = true;
      end
    end
    function posn = roi2RectPos(roi)
      % See notes at top of CropInfo.m re: rois vs posns.
      %
      % roi: [nx4] [xlo xhi ylo yhi], integers
      %
      % posn: [nx4] [xlo ylo w h] imrect-style posn

      posn = [roi(:,1) roi(:,3) roi(:,2)-roi(:,1) roi(:,4)-roi(:,3)];
    end
    function [roi,roiwidth,roiheight] = rectPos2roi(posn)
      % See notes at top of CropInfo.m re: rois vs posns.
      %
      % posn: [nx4] [xlo ylo w h] imrect-style posn
      %
      % roi: [nx4] [xlo xhi ylo yhi], integer. warning thrown if any 
      %   element is <=0.
      % roiwidth: [nx1] horizontal span of roi (includes edges)
      % roiheight: [nx1] vertical "
      %
      % We don't do the smartest possible thing here, the result here may
      % be slightly unexpected at the outer edge of the roi depending on
      % the exact positioning/width/height of posn.
      
      roiwidth = round(posn(:,3))+1;
      roiheight = round(posn(:,4))+1;
      xlo = ceil(posn(:,1));
      ylo = ceil(posn(:,2));
      roi = [xlo xlo+roiwidth-1 ylo ylo+roiheight-1];
      if any(roi(:)<=0)
        warningNoTrace('Non-positive ROI coordinates.');
      end      
    end    
    function posnCtrd = rectPos2Ctrd(posn)
      % posn: [nx4] 
      % posnCtrd: [nx4]      
      posnCtrd = posn;
      posnCtrd(:,[1 2]) = posnCtrd(:,[1 2])+posn(:,[3 4])/2;
    end
    function posn = rectPosCtrd2Pos(posnCtrd)
      % posnCtrd: [nx4]      
      % posn: [nx4] 
      posn = posnCtrd;
      posn(:,[1 2]) = posn(:,[1 2])-posn(:,[3 4])/2;
    end
    function posn = rectPosResize(posn,widthHeight)
      % Alter widthHeight of a set of posn's, keeping the posn's centers
      % fixed. DOES NOT check for any proper-ness.
      %
      % posn: [nx4]
      % widthHeight: [2]
      
      szassert(widthHeight,[1 2]);
      posnCtrd = CropInfo.rectPos2Ctrd(posn);
      posnCtrd(:,[3 4]) = repmat(widthHeight,size(posnCtrd,1),1);
      posn = CropInfo.rectPosCtrd2Pos(posnCtrd);      
    end
    
    function roiPosnCheck
      % invariants
      pos(1,:) = 100*rand(1,4);
      for i=1:2      
        roi(i,:) = CropInfo.rectPos2roi(pos(i,:));
        pos(i+1,:) = CropInfo.roi2RectPos(roi(i,:));
      end
      fprintf('pos:\n');      
      disp(pos);
      fprintf('roi:\n');
      disp(roi);
      
      pos = pos(1,:);
      for i=1:2
        posC(i,:) = CropInfo.rectPos2Ctrd(pos(i,:));
        pos(i+1,:) = CropInfo.rectPosCtrd2Pos(posC(i,:));
      end
      fprintf('pos:\n');
      disp(pos);
      fprintf('posCtred:\n');
      disp(posC);
    end
    
    function xy = roiClipXY(roi,xy)
      % Clip xy coords to a given roi.
      % 
      % roi: [xlo xhi ylo yhi]
      % xy: [nx2]
      
      assert(size(xy,2)==2);
      xy(:,1) = min(max(xy(:,1),roi(1)),roi(2));
      xy(:,2) = min(max(xy(:,2),roi(3)),roi(4));
    end
    
  end
end