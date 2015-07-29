classdef Labeler < handle
% Labeler  Bransonlab Animal Video Labeler
%
% Takes a movie, trx (optional), template (optional), and creates/edits
% "animal model" labels.

  properties (Constant,Hidden)
    VERSION = '0.0';
    DEFAULT_LBLFILENAME = '%s.lbl.mat';
    
    SAVEPROPS = { ...
      'VERSION' 'moviefile' 'nframes' 'trxFilename' 'nTrx' ...
      'labelMode' 'nLabelPoints' 'labelPtsColors' ...
      'labeledpos' 'currFrame' 'currTarget' 'minv' 'maxv'};
    LOADPROPS = {'minv' 'maxv'};
    
    TBLTRX_STATIC_COLSTBL = {'id' 'labeled'};
    TBLTRX_STATIC_COLSTRX = {'id' 'labeled'};
    
    TBLFRAMES_COLS = {'frame' 'labeled targets'};
    
    FRAMEUP_BIGSTEP = 10;
    NEIGHBORING_FRAME_MAXRADIUS = 10;
    NEIGHBORING_FRAME_OFFSETS = neighborIndices(Labeler.NEIGHBORING_FRAME_MAXRADIUS);
  end
  
  
  %% Movie
  properties
    movieReader = [];         % MovieReader object
    minv = 0;
    maxv = inf;
  end
  properties (Dependent)
    hasMovie;
    moviefile;
    nframes;
    movienr;
    movienc;
  end
  
  
  %% Trx
  properties
    trxFilename = '';         % full filename
    trx = [];                 % trx object
    zoomRadiusDefault = 100;  % default zoom box size in pixels
    zoomRadiusTight = 10;     % zoom size on maximum zoom (smallest pixel val)
    frm2trx = [];             % nFrm x nTrx logical. frm2trx(iFrm,iTrx) is true if trx iTrx is live on frame iFrm
  end  
  properties (Dependent)
    hasTrx
    currTrx
    nTrx
    nTargets          % nTrx, or 1 if no Trx
  end
  
  
  %% Labeling
  properties
    %labels = cell(0,1);  % cell vector with nTarget els. labels{iTarget} is nModelPts x 2 x "numFramesTarget"
    nLabelPoints;         % scalar integer
    labelPtsColors;       % nLabelPoints x 3 RGB
    
    labelMode;            % scalar LabelMode
    labeledpos;           % labels, npts x 2 x nFrm x nTrx
    labelsLocked;         % nFrm x nTrx
%     labelNames;           % nLabelPoints-by-1 cellstr
    
    lblCore;
    
    lblPrev_ptsH;         % Maybe encapsulate this and next with axes_prev, image_prev
    lblPrev_ptsTxtH;                          
  end  
  
  properties
    gdata = [];           % handles structure for figure

    currFrame = 1;        % current frame
    currIm = [];
    prevFrame = nan;      % last previously VISITED frame
    prevIm = [];    
    currTarget = nan;
    prevTarget = nan;
  end
  
  methods % dependent prop getters
    function v = get.hasMovie(obj)
      v = obj.movieReader.isOpen;
    end    
    function v = get.moviefile(obj)
      mr = obj.movieReader;
      if isempty(mr)
        v = [];
      else
        v = mr.filename;
      end
    end
    function v = get.movienr(obj)
      mr = obj.movieReader;
      if mr.isOpen
        v = mr.nr;        
      else
        v = [];
      end
    end
    function v = get.movienc(obj)
      mr = obj.movieReader;
      if mr.isOpen
        v = mr.nc;        
      else
        v = [];
      end
    end    
    function v = get.nframes(obj)
      v = obj.movieReader.nframes;
    end      
    function v = get.hasTrx(obj)
      v = ~isempty(obj.trx);
    end
    function v = get.currTrx(obj)
      if obj.hasTrx
        v = obj.trx(obj.currTarget);
      else
        v = [];
      end
    end
    function v = get.nTrx(obj)
      v = numel(obj.trx);
    end
    function v = get.nTargets(obj)
      if obj.hasTrx
        v = obj.nTrx;
      else
        v = 1;
      end
    end
  end
  
  methods % prop access
    function set.labeledpos(obj,v)
      obj.labeledpos = v;
      if obj.hasTrx
        obj.updateTrxTable();
      end
      obj.updateFrameTableIncremental(); % TODO use listener/event for this
    end
  end

  
  %% Save/Load
  methods 
    
    function saveLblFile(obj,fname)
      % Saves a .lbl file. Currently defaults to same dir as moviefile.
      
      if exist('fname','var')==0 && obj.hasMovie
        if ~obj.hasMovie
          % extremely unlikely
          error('Labeler:save','No movie loaded.');          
        end
      
        movieFile = obj.movieReader.filename;
        [moviePath,movieFile] = myfileparts(movieFile);
        defaultFname = sprintf(obj.DEFAULT_LBLFILENAME,movieFile);
        filterspec = fullfile(moviePath,defaultFname);
        
        [fname,pth] = uiputfile(filterspec,'Save label file');
        if isequal(fname,0)
          return;
        end
        fname = fullfile(pth,fname);
      elseif exist(fname,'file')>0
        warning('Labeler:save','Overwriting file ''%s''.',fname);
      end
      
      s = obj.getSaveStruct(); %#ok<NASGU>
      save(fname,'-mat','-struct','s');
      
      RC.saveprop('lastLblFile',fname);
    end
    
    function s = getSaveStruct(obj)
      s = struct();
      s.moviefile = obj.movieReader.filename;
      for f = obj.SAVEPROPS, f=f{1}; %#ok<FXSET>
        s.(f) = obj.(f);
      end
      
      switch obj.labelMode
        case LabelMode.TEMPLATE
          s.template = obj.lblCore.getTemplate();
      end
    end
    
    function loadLblFile(obj,fname)
      % Load a lbl file, along with moviefile and trxfile referenced therein
            
      if exist('fname','var')==0
        lastLblFile = RC.getprop('lastLblFile');
        if isempty(lastLblFile)
          lastLblFile = pwd;
        end
        filterspec = sprintf(obj.DEFAULT_LBLFILENAME,'*');
        [fname,pth] = uigetfile(filterspec,'Load label file',lastLblFile);
        if isequal(fname,0)
          return;
        end
        fname = fullfile(pth,fname);
      end
      
      assert(exist(fname,'file')>0,'File ''%s'' not found.');
      
      s = load(fname,'-mat');
      if ~all(isfield(s,{'VERSION' 'labeledpos'}))
        error('Labeler:load','Unexpected contents in Label file.');
      end
      
      for f = obj.LOADPROPS,f=f{1}; %#ok<FXSET>
        obj.(f) = s.(f);
      end
      
      if isempty(s.trxFilename)
        obj.loadMovie(s.moviefile);
      else
        obj.loadMovie(s.moviefile,s.trxFilename);
      end
                  
      assert(isa(s.labelMode,'LabelMode'));
      obj.labelingInit(s.labelMode,'nPts',s.nLabelPoints,...
        'ptColors',s.labelPtsColors);
      obj.labeledpos = s.labeledpos;
      
      obj.setFrame(s.currFrame);
      obj.setTarget(s.currTarget);
            
      obj.updateFrameTableComplete(); % TODO don't like this, maybe move to UI      
    end
    
  end
  
  
  %% Movie, Trx
  methods
    
    function obj = Labeler(varargin)
      npts = varargin{1};
      
      hFig = LabelerGUI(obj);
      obj.gdata = guidata(hFig);
      
      obj.movieReader = MovieReader;
      
      obj.nLabelPoints = npts;
      obj.labelPtsColors = jet(npts);
    end
    
    function loadMovie(obj,movfile,trxfile)
      % movfile: optional, movie name. If not specified, user will be
      % prompted.
      % trxname: optional, trx filename. If not specified, no trx file will
      % be used. To be prompted to specify a trxfile, specify trxname as [].
      
      if exist('movfile','var')==0 || isempty(movfile)
        lastmov = RC.getprop('lbl_lastmovie');            
        [movfile,movpath] = uigetfile('*.*','Select video to label',lastmov);
        if ~ischar(movfile)
          return;
        end
        movfile = fullfile(movpath,movfile);
      end
      assert(exist(movfile,'file')>0,'File ''%s'' not found.',movfile);

      tfTrx = exist('trxfile','var') > 0;
      if tfTrx
        if isempty(trxfile)
          [trxfile,trxpath] = uigetfile('*.mat','Select trx file',movpath);
          if ~ischar(trxfile)
            % user canceled; interpret this as "there is no trx file"
            tfTrx = false;
          else
            trxfile = fullfile(trxpath,trxfile);
          end
        end
        if tfTrx
          assert(exist(trxfile,'file')>0,'Trx file ''%s'' not found.');
        end
      else
        trxfile = [];
      end
      
      obj.movieReader.open(movfile);
      RC.saveprop('lbl_lastmovie',movfile);
      
      obj.trxFilename = trxfile;

      obj.newMovieAndTrx();
    end
                
  end
  
  
  %% Labeling
  methods 
    
    function labelingInit(obj,labelMode,varargin)
      % Initialize labeling state
      % 
      % Optional PVs:
      % - nPts. Defaults to current nPts
      % - ptNames. Defaults to current
      % - ptColors. Defaults to current

      assert(isa(labelMode,'LabelMode'));
      
      [nPts,ptColors] = myparse(varargin,...
        'nPts',obj.nLabelPoints,...
        'ptColors',obj.labelPtsColors);
      validateattributes(nPts,{'numeric'},{'scalar' 'positive' 'integer'});
     % assert(iscellstr(ptNames) && numel(ptNames)==nPts);
      assert(isequal(size(ptColors),[nPts 3]));
      
      obj.labelMode = labelMode;
      obj.nLabelPoints = nPts;
      obj.labelPtsColors = ptColors;
      
      gd = obj.gdata;
      switch labelMode
        case LabelMode.SEQUENTIAL
          obj.lblCore = LabelCoreSeq(obj);
          gd.menu_setup_sequential_mode.Enable = 'on';
          gd.menu_setup_sequential_mode.Checked = 'on';
          gd.menu_setup_template_mode.Enable = 'off';
          gd.menu_setup_template_mode.Checked = 'on';
          gd.menu_setup_createtemplate.Enable = 'off';
        case LabelMode.TEMPLATE
          obj.lblCore = LabelCoreTemplate(obj);
          gd.menu_setup_sequential_mode.Enable = 'off';
          gd.menu_setup_sequential_mode.Checked = 'off';
          gd.menu_setup_template_mode.Enable = 'on';
          gd.menu_setup_template_mode.Checked = 'on';
          gd.menu_setup_createtemplate.Enable = 'on';
      end
      obj.lblCore.init(nPts,ptColors);
      
      obj.labelPosInitWithLocked();
      
      deleteHandles(obj.lblPrev_ptsH);
      deleteHandles(obj.lblPrev_ptsTxtH);
      obj.lblPrev_ptsH = nan(obj.nLabelPoints,1);
      obj.lblPrev_ptsTxtH = nan(obj.nLabelPoints,1);
      axprev = obj.gdata.axes_prev;
      for i = 1:obj.nLabelPoints
        obj.lblPrev_ptsH(i) = plot(axprev,nan,nan,'w+','MarkerSize',20,...
                                   'LineWidth',3,'Color',ptColors(i,:),'UserData',i);
        obj.lblPrev_ptsTxtH(i) = text(nan,nan,num2str(i),'Parent',axprev,...
                                      'Color',ptColors(i,:),'Hittest','off');
      end      
    end
    
    %%% labelpos
      
    function labelPosInitWithLocked(obj)
      obj.labeledpos = nan(obj.nLabelPoints,2,obj.nframes,obj.nTargets); 
      obj.labelsLocked = false(obj.nframes,obj.nTargets);
    end
    
    function labelPosClear(obj)
      % Clear all labels for current frame, current target
      obj.labeledpos(:,:,obj.currFrame,obj.currTarget) = nan;
    end
    
    function [tf,lpos] = labelPosIsLabeled(obj,iFrm,iTrx)
      lpos = obj.labeledpos(:,:,iFrm,iTrx);
      tfnan = isnan(lpos);
      assert(all(tfnan(:)) || ~any(tfnan(:)));
      tf = ~any(tfnan(:));
    end 
    
    function labelPosSet(obj,xy)
      % Set labelpos from labelPtsH for current frame, current target
      
      assert(~any(isnan(xy(:))));
      
      cfrm = obj.currFrame;
      ctrx = obj.currTarget;
      obj.labeledpos(:,:,cfrm,ctrx) = xy;
    end

    function [tfneighbor,iFrm0,lpos0] = labelPosLabeledNeighbor(obj,iFrm,iTrx)
      % tfneighbor: if true, a labeled neighboring frame was found
      % iFrm0: index of labeled neighboring frame, relevant only if
      %   tfneighbor is true
      % lpos0: labels at iFrm0, relevant only if tfneighbor is true
      %
      % This method looks for a frame "near" iFrm for target iTrx that is
      % labeled. This could be iFrm itself if it is labeled. If a
      % neighboring frame is found, iFrm0 is not guaranteed to be the
      % closest or any particular neighboring frame although this will tend
      % to be true.      
      
      lposTrx = obj.labeledpos(:,:,:,iTrx);
      for dFrm = 0:obj.NEIGHBORING_FRAME_OFFSETS 
        iFrm0 = iFrm + dFrm;
        iFrm0 = max(iFrm0,1);
        iFrm0 = min(iFrm0,obj.nframes);
        lpos0 = lposTrx(:,:,iFrm0);
        if ~isnan(lpos0(1))
          tfneighbor = true;
          return;
        end
      end
      
      tfneighbor = false;
      iFrm0 = nan;
      lpos0 = [];      
    end
       
  end
  
  methods (Access=private)
    
    function labelsUpdateNewFrame(obj)
      if ~isempty(obj.lblCore)
        obj.lblCore.newFrame(obj.prevFrame,obj.currFrame,obj.currTarget);
      end
      obj.labelsPrevUpdate();
    end
    
    function labelsUpdateNewTarget(obj)
      if ~isempty(obj.lblCore)
        obj.lblCore.newTarget(obj.prevTarget,obj.currTarget,obj.currFrame);
      end
      obj.labelsPrevUpdate();
    end
    
    function labelsPrevUpdate(obj)
      if ~isnan(obj.prevFrame) && ~isempty(obj.lblPrev_ptsH)
        lpos = obj.labeledpos(:,:,obj.prevFrame,obj.currTarget);
        LabelCore.assignCoords2Pts(lpos,obj.lblPrev_ptsH,obj.lblPrev_ptsTxtH);
      else
        LabelCore.removePts(obj.lblPrev_ptsH,obj.lblPrev_ptsTxtH);
      end
    end
  
  end
 
  methods (Static)
    
    function uv = transformPtsTrx(uv0,trx0,iFrm0,trx1,iFrm1)
      % uv0: npts x 2 array of points
      % trx0: scalar trx
      % iFrm0: absolute frame number for trx0
      % etc
      %
      % The points uv0 correspond to trx0 @ iFrm0. Compute uv that
      % corresponds to trx1 @ iFrm1, ie so that uv relates to trx1@iFrm1 in 
      % the same way that uv0 relates to trx0@iFrm0.
      
      assert(trx0.off==1-trx0.firstframe);
      assert(trx1.off==1-trx1.firstframe);
      
      iFrm0 = iFrm0+trx0.off;
      xy0 = [trx0.x(iFrm0) trx0.y(iFrm0)];
      th0 = trx0.theta(iFrm0);
      
      iFrm1 = iFrm1+trx1.off;
      xy1 = [trx1.x(iFrm1) trx1.y(iFrm1)];
      th1 = trx1.theta(iFrm1);
      
      uv = transformPoints(uv0,xy0,th0,xy1,th1);
    end
        
  end
  
  
  %% Video
  methods
    
    function videoResetView(obj)
      axis(obj.gdata.axes_curr,'auto','image');
      %axis(obj.gdata.axes_prev,'auto','image');
    end
    
    function videoCenterOnCurrTarget(obj)
      [x0,y0] = obj.videoCurrentCenter;
      [x,y] = obj.currentTargetLoc();
      
      dx = x-x0;
      dy = y-y0;
      axisshift(obj.gdata.axes_curr,dx,dy);
      %axisshift(obj.gdata.axes_prev,dx,dy);
    end
    
    function videoZoom(obj,zoomRadius)
      % Zoom to square window over current frame center with given radius.
      
      [x0,y0] = obj.videoCurrentCenter();      
      lims = [x0-zoomRadius,x0+zoomRadius,y0-zoomRadius,y0+zoomRadius];
      axis(obj.gdata.axes_curr,lims);
      axis(obj.gdata.axes_prev,lims);      
    end
    function videoZoomFac(obj,zoomFac)
      % zoomFac: 0 for no-zoom; 1 for max zoom
      
      zr0 = max(obj.movienr,obj.movienc)/2; % no-zoom: large radius
      zr1 = obj.zoomRadiusTight; % tight zoom: small radius
      
      if zr1>zr0
        zr = zr0;
      else
        zr = zr0 + zoomFac*(zr1-zr0);
      end
      obj.videoZoom(zr);      
    end
    
    function [xsz,ysz] = videoCurrentSize(obj)
      v = axis(obj.gdata.axes_curr);
      xsz = v(2)-v(1);
      ysz = v(4)-v(3);
    end
    function [x0,y0] = videoCurrentCenter(obj)
      v = axis(obj.gdata.axes_curr);
      x0 = mean(v(1:2));
      y0 = mean(v(3:4));
    end
    
    function videoSetContrastFromAxesCurr(obj)
      % Get video contrast from axes_curr and record/set
      clim = get(obj.gdata.axes_curr,'CLim');
      set(obj.gdata.axes_prev,'CLim',clim);
      obj.minv = clim(1);
      obj.maxv = clim(2);
    end
    
  end
  
  
  %%
  methods (Hidden)
    
    function newMovieAndTrx(obj)
      % .movieReader and .trxfilename set

      movRdr = obj.movieReader;
      nframes = movRdr.nframes;

      tfTrx = ~isempty(obj.trxFilename);
      if tfTrx
        tmp = load(obj.trxFilename);
        obj.trx = tmp.trx;
      else
        obj.trx = [];
      end
            
      f2t = false(obj.nframes,obj.nTrx);
      for i = 1:obj.nTrx
        frm0 = obj.trx(i).firstframe;
        frm1 = obj.trx(i).endframe;        
        f2t(frm0:frm1,i) = true;
      end
      obj.frm2trx = f2t;
            
      if obj.hasTrx
        obj.currFrame = min([obj.trx.firstframe]);
      else
        obj.currFrame = 1;
      end
                 
      im1 = movRdr.readframe(obj.currFrame);
      %obj.minv = max(obj.minv,0);
      if isfield(movRdr.info,'bitdepth')
        obj.maxv = min(obj.maxv,2^movRdr.info.bitdepth-1);
      elseif isa(im1,'uint16')
        obj.maxv = min(2^16 - 1,obj.maxv);
      elseif isa(im1,'uint8')
        obj.maxv = min(obj.maxv,2^8 - 1);
      else
        obj.maxv = min(obj.maxv,2^(ceil(log2(max(im1(:)))/8)*8));
      end
      
      %#UI      
      axcurr = obj.gdata.axes_curr;
      axprev = obj.gdata.axes_prev;
      imcurr = obj.gdata.image_curr;
      set(imcurr,'CData',im1);
%       axis(axcurr,'image');
%       axis(axprev,'image');
      set(axcurr,'CLim',[obj.minv,obj.maxv],...
                 'XLim',[.5,size(im1,2)+.5],...
                 'YLim',[.5,size(im1,1)+.5]);
      set(axprev,'CLim',[obj.minv,obj.maxv],...
                 'XLim',[.5,size(im1,2)+.5],...
                 'YLim',[.5,size(im1,1)+.5]);
      zoom(axcurr,'reset');
      zoom(axprev,'reset');
      
      %#UI
      sliderstep = [1/(nframes-1),min(1,100/(nframes-1))];
      set(obj.gdata.slider_frame,'Value',0,'SliderStep',sliderstep);
      
      obj.labelPosInitWithLocked();

      obj.currFrame = 2; % to force update in setFrame
      obj.setTarget(1);
      obj.setFrame(1);
      
      obj.updateFrameTableComplete(); % TODO don't like this, maybe move to UI
   end
    
    function setFrame(obj,frm)
      obj.prevFrame = obj.currFrame;
      obj.prevIm = obj.currIm;
      set(obj.gdata.image_prev,'CData',obj.prevIm);
      set(obj.gdata.txPrevIm,'String',num2str(obj.prevFrame));
      
      if obj.currFrame~=frm
        obj.currIm = obj.movieReader.readframe(frm);
        obj.currFrame = frm;
      end
            
      set(obj.gdata.image_curr,'CData',obj.currIm);
      if obj.hasTrx
        obj.videoCenterOnCurrTarget();
      end
        
      %#UI
      set(obj.gdata.slider_frame,'Value',(frm-1)/(obj.nframes-1));
      set(obj.gdata.edit_frame,'String',num2str(frm));

      obj.labelsUpdateNewFrame();
      
      if obj.hasTrx
        obj.updateTrxTable();
      end
      
      % obj.showPreviousLabels      
      % obj.updateLockedButton(); %#UI
      
%       if obj.currFrame > 1
%         for i = 1:obj.npoints
%           if numel(handles.posprev) < i || ~ishandle(handles.posprev(i)),
%             handles.posprev(i) = plot(handles.axes_prev,nan,nan,'+','Color',handles.templatecolors(i,:),'MarkerSize',8);
%           end
%           set(handles.posprev(i),'XData',handles.labeledpos(i,1,handles.f-1,handles.animal),...
%             'YData',handles.labeledpos(i,2,handles.f-1,handles.animal));
%         end
%       else
%         set(handles.posprev,'XData',nan,'YData',nan);
%       end
      
%       if ~isempty(handles.trx) && ~isempty(handles.template),
%         pushbutton_template_Callback(hObject,[],handles);
%       end
    end
    
    function setTarget(obj,iTgt)
      validateattributes(iTgt,{'numeric'},{'positive' 'integer' '<=' obj.nTargets});
      
      obj.prevTarget = obj.currTarget;
      obj.currTarget = iTgt;
      if obj.hasTrx
        obj.videoCenterOnCurrTarget();
      end
      obj.labelsUpdateNewTarget();
    end
    
    function frameUp(obj,tfBigstep)
      if tfBigstep
        df = obj.FRAMEUP_BIGSTEP;
      else
        df = 1;
      end
      f = min(obj.currFrame+df,obj.nframes);
      obj.setFrame(f);
    end
    
    function frameDown(obj,tfBigstep)
      if tfBigstep
        df = obj.FRAMEUP_BIGSTEP;
      else
        df = 1;
      end
      f = max(obj.currFrame-df,1);
      obj.setFrame(f);
    end
    
%     function clearTarget(obj)
%       obj.currTarget = nan;      
%       obj.videoZoomFac(0);
%     end
    
    function [x,y] = currentTargetLoc(obj)
      if obj.hasTrx
        cfrm = obj.currFrame;
        ctrx = obj.currTrx;

        if cfrm < ctrx.firstframe || cfrm > ctrx.endframe
          warning('Labeler:target','No track for current target at frame %d.',cfrm);
          x = nan;
          y = nan;
        else
          i = cfrm - ctrx.firstframe + 1;
          x = ctrx.x(i);
          y = ctrx.y(i);
        end
      else
        x = round(obj.movienc/2);
        y = round(obj.movienr/2);
      end
    end
                
    % TODO prob use listener/event for this; maintain relevant
    % datastructure in Labeler
    function updateTrxTable(obj)
      % based on .currFrame, .labeledpos
      
      colnames = obj.TBLTRX_STATIC_COLSTRX(1:end-1);

      tfLive = obj.frm2trx(obj.currFrame,:);
      trxLive = obj.trx(tfLive);
      trxLive = trxLive(:);
      trxLive = rmfield(trxLive,setdiff(fieldnames(trxLive),colnames));
      trxLive = orderfields(trxLive,colnames);
      tbldat = struct2cell(trxLive)';
      
      iTrxLive = find(tfLive);
      tfLbled = false(size(iTrxLive(:)));
      lpos = obj.labeledpos;
      cfrm = obj.currFrame;
      for iTrx = iTrxLive(:)'
        tfLbled(iTrx) = any(lpos(:,1,cfrm,iTrx));
      end
      tbldat(:,end+1) = num2cell(tfLbled);
      
      tbl = obj.gdata.tblTrx;
      set(tbl,'Data',tbldat);
    end
    
    % TODO Prob use listener/event for this
    function updateFrameTableIncremental(obj)
      % assumes .labelops and tblFrames differ at .currFrame at most
      %
      % might be unnecessary/premature optim
      
      tbl = obj.gdata.tblFrames;
      dat = get(tbl,'Data');
      frames = cell2mat(dat(:,1));
      
      cfrm = obj.currFrame;
      lpos = obj.labeledpos;
      tfLbled = ~isnan(squeeze(lpos(1,1,cfrm,:)));
      nLbled = nnz(tfLbled);
      
      i = frames==cfrm;
      if nLbled>0
        if any(i)
          assert(nnz(i)==1);
          dat{i,2} = nLbled;
        else
          dat(end+1,:) = {cfrm nLbled};
          [~,idx] = sort(cell2mat(dat(:,1)));
          dat = dat(idx,:);
        end
        set(tbl,'Data',dat);
      else
        if any(i)
          assert(nnz(i)==1);
          dat(i,:) = [];
          set(tbl,'Data',dat);
        end
      end
    end    
    function updateFrameTableComplete(obj)
      lpos = obj.labeledpos;
      tf = ~isnan(squeeze(lpos(1,1,:,:)));
      assert(isequal(size(tf),[obj.nframes obj.nTargets]));
      nLbled = sum(tf,2);
      iFrm = find(nLbled>0);
      nLbled = nLbled(iFrm);
      dat = [num2cell(iFrm) num2cell(nLbled)];

      tbl = obj.gdata.tblFrames;
      set(tbl,'Data',dat);
    end
   
    %#UI No really
%     function updateLockedButton(obj)
%       disp('UPDATELOCK TODO');
%       btn = obj.gdata.togglebutton_lock;
%       if obj.labelsLocked(obj.currFrame,obj.currTarget)
%         set(btn,'BackgroundColor',[.6,0,0],'String','Locked','Value',1);
%       else
%         set(btn,'BackgroundColor',[0,.6,0],'String','Unlocked','Value',0);
%       end
%       setButtonImage(btn);
%     end
    
%     function showPreviousLabels(obj)
      % TODO
%       if isempty(handles.labeledpos),
%         fprev = [];
%       else
%         fprev = find(~isnan(handles.labeledpos(1,1,1:handles.f,handles.animal)),1,'last');
%       end
%       
%       for i = 1:handles.npoints,
%         if numel(handles.hpoly) < i || ~ishandle(handles.hpoly(i)),
%           handles.hpoly(i) = plot(handles.axes_curr,nan,nan,'w+','MarkerSize',20,'LineWidth',3);
%           set(handles.hpoly(i),'Color',handles.templatecolors(i,:),...
%             'ButtonDownFcn',@(hObject,eventdata) PointButtonDownCallback(hObject,eventdata,handles.figure,i));
%           handles.htext(i) = text(nan,nan,num2str(i),'Parent',handles.axes_curr,...
%             'Color',handles.templatecolors);
%         end
%         
%         %if all(~isnan(handles.labeledpos(i,:,handles.f,handles.animal))),
%         if ~isempty(fprev),
%           set(handles.hpoly(i),'XData',handles.labeledpos(i,1,fprev,handles.animal),...
%             'YData',handles.labeledpos(i,2,fprev,handles.animal),'Visible','on');
%           
%           tpos = [handles.labeledpos(i,1,fprev,handles.animal)+handles.dt2p;...
%             handles.labeledpos(i,2,fprev,handles.animal)];
%           set(handles.htext(i),'Position',tpos,'Visible','on');          
%         end        
%       end      
%     end   
    
  end

end
