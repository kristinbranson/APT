classdef RigView < handle

  properties (Constant,Hidden)
    AXTAGS = {'axes1' 'axes2' 'axes3'};
    AXCAMS = {'l' 'r' 'b'};
    CAM2IDX = struct('l',1,'r',2,'b',3);
  end
  
  properties
    gdata
    crig
    
    movRdrs % [3] array movieReaders (L,R,B)
  end
  properties (SetObservable)
    movDir
    movCurrFrameLR % current L/R frame
  end
  properties    
    % alternative image source: image stacks
    imDir
    imCurrImIdx
    tfImMode = false; % if true, we are using image stacks source
  end
  
  properties (Hidden)
    anns  % Column vec of RigViewAnns
    annCMap % [nClrx3] colormap for anns
  end  
  
  methods
    function obj = RigView(crFname)
      % crFname: filename containing CalibratedRig object
      
      % load the CalibratedRig
      if exist('crFname','var')==0
        last = RC.getprop('lastCalibratedRigFilename');
        if ~isempty(last)
          [fname,pname] = uigetfile('*.mat','Specify Calibrated Rig mat file',last);
        else
          [fname,pname] = uigetfile('*.mat','Specify Calibrated Rig mat file');
        end
        if isequal(fname,0)
          error('RigView:ctor','No Calibrated Rig specified.');
        end
        crFname = fullfile(pname,fname);
      end
      calrig = load(crFname);
      RC.saveprop('lastCalibratedRigFilename',crFname);
      fn = fieldnames(calrig);
      if ~isscalar(fn)
        error('RigView:ctor',...
          'Calibrated Rig MAT-file expected to contain a single variable (the CalibratedRig object).');
      end
      calrig = calrig.(fn{1});          
      
      hFig = RigViewGUI(obj);
      gd = guidata(hFig);
      obj.gdata = gd;
      obj.crig = calrig;
      
      for iAx = 1:3
        axTag = RigView.AXTAGS{iAx};      
        gd.(axTag).UserData = iAx;
        gd.(axTag).ButtonDownFcn = @(src,evt)obj.axBDF(src,evt);
      end
      
      for i=3:-1:1
        mr(i,1) = MovieReader();
        mr(i).forceGrayscale = 1;
      end
      obj.movRdrs = mr;
        
      obj.anns = RigViewAnn.empty(0,1);
      cm = jet(7);
      cm = cm(3:end,:);
      obj.annCMap = cm;
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
        switch i
          case {1 2}
            imsmall = imadjust(imsmall,[0 0.2],[]);
          case 3
            imsmall = imadjust(imsmall,[0 0.04],[]);
        end
        
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
      
      obj.movCurrFrameLR = frmLR;
    end
    function movFrameUp(obj)
      f = obj.movCurrFrameLR;
      maxFrameLR = min([obj.movRdrs(1:2).nframes]);
      if f+2<=maxFrameLR
        obj.movSetFrameLR(f+2);
      end
    end
    function movFrameDown(obj)
      f = obj.movCurrFrameLR;
      if f-2>=3
        obj.movSetFrameLR(f-2);
      end      
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
      
%     function selectAx(obj,iAx)
%       obj.tfAxSel(iAx) = true;
%       obj.initIMPsLines();
%     end
%     function unselectAxRaw(obj,iAx)
%       obj.tfAxSel(iAx) = false;
%       %obj.initIMPsLines();
%     end
%     function initIMPsLines(obj)
%       for iAx = 1:3
%         imps = obj.hIMPs{iAx};
%         for iIMP = 1:numel(imps)
%           delete(imps(iIMP));
%         end
%       end
%       cellfun(@deleteValidGraphicsHandles,obj.hLines);
%       h = IMP.empty(1,0);
%       obj.hIMPs = repmat({h},3,1);
%       h = gobjects(0,2);
%       obj.hLines = repmat({h},3,1);
%       
% %       delete(obj.hIMPs4);
% %       obj.hIMPs4 = gobjects(1,0);      
% %       delete(obj.hLines4);
% %       obj.hLines4 = gobjects(0,1);
%     end
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
    
    function axBDF(obj,hAx,evt)
      iAx = hAx.UserData;
      switch evt.Button
        case 1
          xy = hAx.CurrentPoint(1,1:2);
          hIMP = IMP(obj.gdata.axs(iAx),xy);
          
          nAnn = numel(obj.anns);
          cmap = obj.annCMap;
          nclr = size(cmap,1);
          cIdx = mod(nAnn,nclr)+1;
          clr = cmap(cIdx,:);
          
          rva = RigViewAnn(obj.gdata.axs,clr,iAx,hIMP);
          obj.anns(end+1,1) = rva;

          hIMP.setColor(clr);
          hIMP.addNewPositionCallback(@(pos)obj.updateEPLines(rva,pos));
          hIMP.deleteFcn = @()obj.rmAnn(rva);
     
          obj.updateEPLines(rva,xy);
          
          fcn = @(src,evt)obj.epiBDF(src,evt);
          rva.epiHLines(1).ButtonDownFcn = fcn;
          rva.epiHLines(2).ButtonDownFcn = fcn;
        case 2
          % none; need to right-click on line          
      end      
    end
    
    function epiBDF(obj,hLine,evt)
      switch evt.Button
        case 3
          ax2 = hLine.Parent;
          iAx2 = find(obj.gdata.axs==ax2);
          
          xy = ax2.CurrentPoint(1,1:2);
          hAnn = hLine.UserData;
          
          hIMP = IMP(ax2,xy);
          hIMP.addNewPositionCallback(@(pos)obj.updateReconstructedPt(hAnn,pos));
          hAnn.addSecondPt(iAx2,hIMP); %#ok<FNDSB>
          
          obj.updateReconstructedPt(hAnn,xy);
      end
    end
        
    function rmAnn(obj,hAnn)
      tf = hAnn==obj.anns;
      assert(nnz(tf)==1);
      delete(obj.anns(tf));
      obj.anns(tf,:) = [];
    end
    
    function updateEPLines(obj,hAnn,pos)
      % Update epipolar lines for given Ann 
      % 
      % hAnn: scalar RigViewAnn handle
      % pos: [2] (x,y) vector, pixel position in axis hAnn.anchorIAx (Cropped coords)
      
      assert(numel(pos)==2);
      %fprintf(1,'Cam %d: croppedcoords: %s\n',iAx,mat2str(round(pos(:)')));
      
      iAx1 = hAnn.anchorIAx;
      iAx2 = hAnn.epiIAx(1);
      iAx3 = hAnn.epiIAx(2);
      cam1 = RigView.AXCAMS{iAx1};
      cam2 = RigView.AXCAMS{iAx2};
      cam3 = RigView.AXCAMS{iAx3};
      rig = obj.crig;
      
      if obj.tfImMode
        xp = [pos(1)-1 pos(2)-1]'; % 0-based  
      else % pos is cropped coords
        y = [pos(2) pos(1)];
        xp = rig.y2x(y,cam1);
      end
      assert(isequal(size(xp),[2 1]));
      xn1 = rig.projected2normalized(xp,cam1);
      
      % create 3D segment by projecting normalized coords into 3D space
      % (coord sys of cam1)
      Zc1 = 0:.25:100; % mm
      Xc1 = [xn1(1)*Zc1; xn1(2)*Zc1; Zc1];
      
      Xc2 = rig.camxform(Xc1,[cam1 cam2]); % 3D seg, in frame of cam2
      Xc3 = rig.camxform(Xc1,[cam1 cam3]); % etc, cam3
      xn2 = [Xc2(1,:)./Xc2(3,:); Xc2(2,:)./Xc2(3,:)]; % normalize
      xn3 = [Xc3(1,:)./Xc3(3,:); Xc3(2,:)./Xc3(3,:)];      
      xp2 = rig.normalized2projected(xn2,cam2); % project
      xp3 = rig.normalized2projected(xn3,cam3);
      if obj.tfImMode
%         r1 = xp2(2,:)+1;
%         c1 = xp2(1,:)+1;
%         tfOOB = r1<1 | r1>1200 | c1<1 | c1>1920;
%         r1 = r1(~tfOOB);
%         c1 = c1(~tfOOB);
% 
%         r2 = xp3(2,:)+1;
%         c2 = xp3(1,:)+1;    
%         tfOOB = r2<1 | r2>1200 | c2<1 | c2>1920;
%         r2 = r2(~tfOOB);
%         c2 = c2(~tfOOB);
      else
        y2 = rig.x2y(xp2,cam2);
        y3 = rig.x2y(xp3,cam3);
        y2 = obj.cropLines(y2,cam2);
        y3 = obj.cropLines(y3,cam3);
        r2 = y2(:,1);
        c2 = y2(:,2);
        r3 = y3(:,1);
        c3 = y3(:,2);        
      end
      hLns = hAnn.epiHLines;
      hLns(1).XData = c2;
      hLns(1).YData = r2;
      hLns(2).XData = c3;
      hLns(2).YData = r3;
      
      
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
    
    function updateReconstructedPt(obj,hAnn,pos2)
      % hAnn: scalar RigViewAnn
      % pos2: xy of point 2 for hAnn
      
      assert(numel(pos2)==2);
      
      iAx1 = hAnn.anchorIAx;
      iAx2 = hAnn.secondIAx;
      iAx3 = hAnn.thirdIAx; % reconstruct pt in this axis
      cam1 = RigView.AXCAMS{iAx1};
      cam2 = RigView.AXCAMS{iAx2};
      cam3 = RigView.AXCAMS{iAx3};
      rig = obj.crig;

      assert(~obj.tfImMode);
      
      % get projected pts for 1 (anchor) and 2 (second)
      pos1 = hAnn.anchorHIMP.getPosition;
      y1 = [pos1(2) pos1(1)];
      y2 = [pos2(2) pos2(1)];
      xp1 = rig.y2x(y1,cam1);
      xp2 = rig.y2x(y2,cam2);
      assert(isequal(size(xp1),size(xp2),[2 1]));
      
      [X1,X2,d,P,Q] = rig.stereoTriangulate(xp1,xp2,cam1,cam2);
      % X1: [3x1]. 3D coords in frame of camera1
      % X2: etc
      % d: error/discrepancy in closest approach
      % P: 3D point of closest approach on normalized ray of camera 1, in
      % frame of camera 2
      % Q: 3D point of closest approach on normalized ray of camera 2, in
      % frame of camera 2
      
      X3 = rig.camxform(X2,[cam2 cam3]);
      P3 = rig.camxform(P,[cam2 cam3]);
      Q3 = rig.camxform(Q,[cam2 cam3]);
      
      xp3 = rig.project(X3,cam3);
      pp3 = rig.project(P3,cam3);
      qp3 = rig.project(Q3,cam3);
      yx3 = rig.x2y(xp3,cam3);
      yp3 = rig.x2y(pp3,cam3);
      yq3 = rig.x2y(qp3,cam3);
      assert(isequal(size(yx3),size(yp3),size(yq3),[1 2])); % [row col]
      
      hLine = hAnn.thirdHPt;
      hLine.XData = [yp3(2) yx3(2) yq3(2)];
      hLine.YData = [yp3(1) yx3(1) yq3(1)];
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