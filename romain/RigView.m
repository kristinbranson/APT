classdef RigView < handle

  properties (Constant,Hidden)
    AXTAGS = {'axes1' 'axes2' 'axes3'};
    AXCAMS = {'l' 'r' 'b'};
    CAM2IDX = struct('l',1,'r',2,'b',3);
  end
  
  properties
    gdata
    crig
    
    movDir
    movRdrs % [3] array movieReaders (L,R,B)
    movCurrFrameLR % current L/R frame
    
    imDir
    imCurrImIdx
    tfImMode = false;
  end
  properties (SetObservable)
    tfAxSel % [3] logical array, flags for selected axes 
  end 
  properties (Hidden)
    hIMPs  % [3] cell array of handle vecs of current impoints for axes 1-3
    
    % [3] cell array of handle vecs of current epipolar lines correponding 
    % to hIMPs. hLines{iAx} is [numel(hIMPs{iAx})x2] array;
    % hLines{iAx}(iIMP,:) are two handles to lines in the other two axes
    hLines 

    impCMap % [Nx3] colormap for IMPs/IMPELs (IMP epipolar lines)
%     hIMPs4
%     hLines4
  end
  properties (Dependent)
    nSel % scalar, number of selected axes
%     iAxSel
%     iAxNotSel
  end
  
  methods % Dep prop getters
    function v = get.nSel(obj)
      v = nnz(obj.tfAxSel);
    end
%     function v = get.iAxNotSel(obj)
%       v = find(~obj.tfAxSel);
%       v = v(:);
%     end
  end    
  
  methods
    function obj = RigView
      hFig = RigViewGUI(obj);
      gd = guidata(hFig);
      obj.gdata = gd;      
      
      for iAx = 1:3
        axTag = RigView.AXTAGS{iAx};      
        gd.(axTag).UserData = iAx;
        gd.(axTag).ButtonDownFcn = @(src,evt)obj.axBDF(src);
      end
      
      for i=3:-1:1
        mr(i,1) = MovieReader();
        mr(i).forceGrayscale = 1;
      end
      obj.movRdrs = mr;
        
      obj.tfAxSel = false(3,1);
      h = IMP.empty(1,0);
      obj.hIMPs = repmat({h},3,1);
      h = gobjects(0,2);
      obj.hLines = repmat({h},3,1);
      cm = parula(7);
      obj.impCMap = [[1 0 0];cm([2 3 5 6],:)]; % brighter colors
    end
    
    function delete(obj) %#ok<INUSD>
      % none
    end    
  end
  
  methods
    
    function movLoadMovs(obj,movdir)
      % movdir: directory containing 3 movies
      
      dd = dir(movdir);
      names = {dd.name}';
      tfB = cellfun(@(x)~isempty(regexp(x,'_cam_2_','once')),names);
      tfL = cellfun(@(x)~isempty(regexp(x,'_cam_0_','once')),names);
      tfR = cellfun(@(x)~isempty(regexp(x,'_cam_1_','once')),names);
      assert(nnz(tfB)==1);
      assert(nnz(tfL)==1);
      assert(nnz(tfR)==1);
      
      mrs = obj.movRdrs;
      mrs(1).open(fullfile(movdir,names{tfL}));
      mrs(2).open(fullfile(movdir,names{tfR}));
      mrs(3).open(fullfile(movdir,names{tfB}));
      obj.movDir = movdir;
      
      obj.movSetFrameLR(3);      
    end
    function movSetFrameLR(obj,frmLR)
      if mod(frmLR,2)==0
        error('RigView:frmLR','frmLR must be odd.');
      end
      frmB = (frmLR-1)/2;
      
      mrs = obj.movRdrs;
      im = {mrs(1).readframe(frmLR); mrs(2).readframe(frmLR); mrs(3).readframe(frmB)};
      
      hIm = obj.gdata.ims;
      rig = obj.crig;
      for i=1:3
        % pad image
        imfull = zeros(1200,1920);
        cam = obj.AXCAMS{i};
        roi = rig.roi.(cam);
        rowIdx = roi.offsetY+1:roi.offsetY+roi.height;
        colIdx = roi.offsetX+1:roi.offsetX+roi.width;
        
        % contrast
        imsmall = im{i};
        imsmall = uint8(imsmall);
        imsmall = imadjust(imsmall,[0 0.2],[]);
        
        % outline
        BORDERWIDTH = 2; 
        imsmallborder = imsmall;
        imsmallborder(1:BORDERWIDTH,:) = 255;
        imsmallborder(end-BORDERWIDTH+1:end,:) = 255;
        imsmallborder(:,1:BORDERWIDTH) = 255;
        imsmallborder(:,end-BORDERWIDTH+1:end) = 255;        
        imfull(rowIdx,colIdx) = imsmallborder;
        
        if i==3
          imfull = flipud(imfull);
        end
        
        %hIm(i).CData = imfull;
        hIm(i).CData = imsmall;
      end
      
%       hIm(4).CData = imsmall;
      
      obj.movCurrFrameLR = frmLR;
    end
    
    function imSetImDir(obj,dir)
      obj.imDir = dir;      
    end
    function imSetCurrIm(obj,idx)
      hIm = obj.gdata.ims;
      for iCam=1:3
        im = imread(fullfile(obj.imDir,...
          sprintf('cam%d',iCam-1),...
          sprintf('cam%d-%02d.jpg',iCam-1,idx)));
        hIm(iCam).CData = im;
      end
      obj.imCurrImIdx = idx;
      
      obj.tfImMode = true;
    end
      
    function selectAx(obj,iAx)
      obj.tfAxSel(iAx) = true;
      obj.initIMPsLines();
    end
    function unselectAxRaw(obj,iAx)
      obj.tfAxSel(iAx) = false;
      %obj.initIMPsLines();
    end
    function initIMPsLines(obj)
      for iAx = 1:3
        imps = obj.hIMPs{iAx};
        for iIMP = 1:numel(imps)
          delete(imps(iIMP));
        end
      end
      cellfun(@deleteValidHandles,obj.hLines);
      h = IMP.empty(1,0);
      obj.hIMPs = repmat({h},3,1);
      h = gobjects(0,2);
      obj.hLines = repmat({h},3,1);
      
%       delete(obj.hIMPs4);
%       obj.hIMPs4 = gobjects(1,0);      
%       delete(obj.hLines4);
%       obj.hLines4 = gobjects(0,1);
    end
  end
    
    % UI
    %
    % - Toggling TBs will alter selected axes. This will show/hide
    % hPts/hLines and go from epipolar to stereo mode etc.
    %
    % - Epipolar mode. Clicking on the selected axes in a fresh location
    % creates an impoint. The epipolar lines corresponding to this pt are
    % shown in the other two axes, with appropriate color-coding. Dragging
    % an existing impoint live-updates its epipolar lines. Multiple pts may
    % be created. Pts may be deleted by right-click using the regular
    % impoint UI.
    %
    % - Stereo mode. Clicking on a selected axis creates an impoint, if one
    % does not exist. If an impoint exists, nothing happens, or maybe its 
    % location is updated. When impoints exist in both selected axes, the
    % calibrated projection for the 3rd view is shown as a single pt.
    % Dragging an impoint in an selected axis live-updates the pt in the
    % 3rd view.
    
  methods
    
    function axBDF(obj,hAx)
      iAx = hAx.UserData;
      if ~obj.tfAxSel(iAx)
        if obj.nSel==1
          obj.selectAx(iAx);
        else % nSel==2
          iAxOther = setdiff(1:3,iAx);
          arrayfun(@(x)obj.unselectAxRaw(x),iAxOther);
          obj.selectAx(iAx);          
        end
      else
        xy = hAx.CurrentPoint(1,1:2);
        iIMP = obj.createIMP(iAx,xy);
        obj.updateIMPEPLines(iAx,iIMP,xy);
      end      
    end
    
    function iIMP = createIMP(obj,iAx,xy)
      % Create/add a new impoint for axis iAx.
      %
      % xy: [2] vec of coords at which to create new IMP (cropped coords)
      
      assert(numel(obj.hIMPs{iAx})==size(obj.hLines{iAx},1));
      
      axs = obj.gdata.axs;
      h = IMP(axs(iAx),xy);      
      
      obj.hIMPs{iAx}(end+1) = h;
      iIMP = numel(obj.hIMPs{iAx});
      cm = obj.impCMap;
      nClr = size(cm,1);
      iClr = mod(iIMP-1,nClr)+1;
      clr = cm(iClr,:);
      h.setColor(clr);
      h.addNewPositionCallback(@(pos)obj.updateIMPEPLines(iAx,iIMP,pos));
      h.deleteFcn = @()obj.rmIMP(iAx,iIMP);
      
      iAx1 = mod(iAx,3)+1; 
      iAx2 = mod(iAx+1,3)+1;
      obj.hLines{iAx}(end+1,:) = ...
        [plot(axs(iAx1),nan,nan,'Color',clr) ...
         plot(axs(iAx2),nan,nan,'Color',clr)];
       
%       obj.hLines4(end+1,:) = plot(axs(4),nan,nan,'Color',clr);
%       hold(axs(4),'on');
       
      hold(axs(iAx1),'on');
      hold(axs(iAx2),'on');
     
      obj.updateIMPEPLines(iAx,iIMP,xy);
    end
    
    function rmIMP(obj,iAx,iIMP)
      delete(obj.hIMPs{iAx}(iIMP));
      delete(obj.hLines{iAx}(iIMP));
      obj.hIMPs{iAx}(:,iIMP) = [];
      obj.hLines{iAx}(:,iIMP) = [];
    end
    
    function updateIMPEPLines(obj,iAx,iIMP,pos)
      % Update epipolar lines for given IMP in iAx
      % 
      % iAx: axis/camera containing IMP
      % iIMP: which IMP
      % pos: [2] (x,y) vector, pixel position in axis iAx (Cropped coords)
      
      assert(numel(pos)==2);
      %fprintf(1,'Cam %d: croppedcoords: %s\n',iAx,mat2str(round(pos(:)')));
      
      iAx1 = mod(iAx,3)+1; 
      iAx2 = mod(iAx+1,3)+1;
      cam = RigView.AXCAMS{iAx};
      cam1 = RigView.AXCAMS{iAx1};
      cam2 = RigView.AXCAMS{iAx2};
      rig = obj.crig;
      
      if obj.tfImMode
        xp = [pos(1)-1 pos(2)-1]'; % 0-based  
      else % pos is cropped coords
        y = [pos(2) pos(1)];
        xp = rig.y2x(y,cam);
      end
      assert(isequal(size(xp),[2 1]));
      xn = rig.projected2normalized(xp,cam);
      
      % create 3D segment by projecting normalized coords into 3D space
      % (coord sys of 'cam')
      Zc = 0:100; % mm
      Xc = [xn(1)*Zc; xn(2)*Zc; Zc];
      
      Xc1 = rig.camxform(Xc,[cam cam1]); % 3D seg, in frame of cam1
      Xc2 = rig.camxform(Xc,[cam cam2]); % etc, cam2      
      xn1 = [Xc1(1,:)./Xc1(3,:); Xc1(2,:)./Xc1(3,:)]; % normalize
      xn2 = [Xc2(1,:)./Xc2(3,:); Xc2(2,:)./Xc2(3,:)];      
      xp1 = rig.normalized2projected(xn1,cam1); % project
      xp2 = rig.normalized2projected(xn2,cam2);
      if obj.tfImMode
        r1 = xp1(2,:)+1;
        c1 = xp1(1,:)+1;
        tfOOB = r1<1 | r1>1200 | c1<1 | c1>1920;
        r1 = r1(~tfOOB);
        c1 = c1(~tfOOB);

        r2 = xp2(2,:)+1;
        c2 = xp2(1,:)+1;    
        tfOOB = r2<1 | r2>1200 | c2<1 | c2>1920;
        r2 = r2(~tfOOB);
        c2 = c2(~tfOOB);
      else
        y1 = rig.x2y(xp1,cam1);
        y2 = rig.x2y(xp2,cam2);
        y1 = obj.cropLines(y1,cam1);
        y2 = obj.cropLines(y2,cam2);
        r1 = y1(:,1);
        c1 = y1(:,2);
        r2 = y2(:,1);
        c2 = y2(:,2);        
      end
      hLns = obj.hLines{iAx}(iIMP,:);
      hLns(1).XData = c1;
      hLns(1).YData = r1;
      hLns(2).XData = c2;
      hLns(2).YData = r2;
      
      
%       if iAx1==3 || iAx2==3
%         if iAx1==3
%           y4 = rig.x2y(xp1,'b');
%         else
%           y4 = rig.x2y(xp2,'b');  
%         end
%         y4 = obj.cropLines(y4,'b');  
%         obj.hLines4(1).XData = y4(:,2);
%         obj.hLines4(1).YData = y4(:,1);
%       end
    end
    
    function y = cropLines(obj,y,cam)
      % "Crop" lines projected on image -- replace points that lie outside
      % of image with NaN.
      %
      % y: [Nx2] (row,col) cropped coords
      % cam: 'l','r','b'
      %
      % y: [Nx2], with OOB points replaced with nan in both coords
      
      assert(size(y,2)==2);
      
      idx = obj.CAM2IDX.(cam);
      mr = obj.movRdrs(idx);
      nr = mr.nr;
      nc = mr.nc;
      
      rows = y(:,1);
      cols = y(:,2);
      tfOOB = rows<1 | rows>nr | cols<1 | cols>nc;
      y(tfOOB,:) = nan;      
    end
    
  end
  
end