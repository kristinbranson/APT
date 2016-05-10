classdef RigView < handle

  properties (Constant,Hidden)
    AXTAGS = {'axes1' 'axes2' 'axes3'};
  end
  
  properties
    gdata
    crig
    
    movDir
    movRdrs % [3] array movieReaders (L,R,B)
    movCurrFrameLR % current L/R frame 
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
  end
  properties (Dependent)
    nSel % scalar, number of selected axes
%     iAxSel
%     iAxNotSel
  end
  
  methods 
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
      h = gobjects(2,0);
      obj.hLines = repmat({h},3,1);
      cm = parula(7);
      obj.impCMap = cm([2 3 5 6],:); % brighter colors
    end
    
    function delete(obj) %#ok<INUSD>
      % none
    end    
  end
  
  methods    
    function loadMovs(obj,movdir)
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
      
      obj.setFrameLR(3);      
    end
    function setFrameLR(obj,frmLR)
      if mod(frmLR,2)==0
        error('RigView:frmLR','frmLR must be odd.');
      end
      frmB = (frmLR-1)/2;
      
      mrs = obj.movRdrs;
      im = {mrs(1).readframe(frmLR); mrs(2).readframe(frmLR); mrs(3).readframe(frmB)};
      
      %axs = obj.gdata.axs;
      hIm = obj.gdata.ims;
      for i=1:3
        hIm(i).CData = im{i};
      end
      obj.movCurrFrameLR = frmLR;
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
      h = gobjects(2,0);
      obj.hLines = repmat({h},3,1);
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
        iIMP = obj.createIMP(iAx,hAx.CurrentPoint(1,1:2));
        obj.updateIMPEPLines(iAx,iIMP);
      end      
    end
    
    function iIMP = createIMP(obj,iAx,xy)
      % Create/add a new impoint for axis iAx.
      %
      % xy: [2] vec of coords at which to create new IMP
      
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
     
      obj.updateIMPEPLines(iAx,iIMP,xy);
    end
    
    function rmIMP(obj,iAx,iIMP)
      delete(obj.hIMPs{iAx}(iIMP));
      delete(obj.hLines{iAx}(iIMP));
      obj.hIMPs{iAx}(:,iIMP) = [];
      obj.hLines{iAx}(:,iIMP) = [];
    end
    
    function updateIMPEPLines(obj,iAx,iIMP,pos)
      % Update epipolar line for given IMP in iAx
      
    end
           
  end
  
end