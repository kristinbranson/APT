classdef SpecifyMovieToTrackGUI < handle
  
  properties 
    defaultdir = '';
    lObj = [];
    hParent = [];
    movdata = [];
    nview = 1;
    hastrx = false;
    iscrop = false;
    isma = false;
    docalibrate = false;
    nfields = nan;
    gdata = [];
    posinfo = struct;
    colorinfo = struct;
    rowinfo = struct;
    isgood = struct;
    cropwh = [];
    movieReader = {};
    dostore = false;
    defaulttrkpat = [];
    defaulttrxpat = [];
%    defaultdetectpat = [];
    % track_type = 'track';
    detailed_options = true;
    link_type = 'motion';
    showPathEnds = true;  % Path display mode: true = show path ends, false = show path starts
  end
  methods
    function obj = SpecifyMovieToTrackGUI(lObj,hParent,movdata,varargin)
      
      [defaulttrkpat,defaulttrxpat,detailed_options] = myparse(varargin,...
        'defaulttrkpat',[], ... % eg '$movdir/$movfile_$projfile_$trackertype'
        'defaulttrxpat',[], ...
        'detailed_options', true ...
        );

      obj.lObj = lObj;
      obj.isma = lObj.maIsMA;
      obj.hParent = hParent;
      obj.detailed_options = detailed_options;

      obj.nview = obj.lObj.nview;
      obj.hastrx = obj.lObj.hasTrx;
      obj.iscrop = ~obj.hastrx;
      %obj.iscrop = obj.lObj.cropProjHasCrops; % to do: allow cropping when trained without cropping?
      if obj.nview > 1 
        prms = obj.lObj.trackParams ;
        docalibrate = ~strcmpi(prms.ROOT.PostProcess.reconcile3dType, 'none') ;
      else
        docalibrate = false ;
      end
      obj.docalibrate = docalibrate ;
      if obj.iscrop,
        if obj.lObj.cropProjHasCrops,
          obj.cropwh = obj.lObj.cropGetCurrentCropWidthHeightOrDefault();
        else
          [minnc,minnr] = obj.lObj.getMinMovieWidthHeight();
          obj.cropwh = [minnc(:)-1,minnr(:)-1];
        end
      end

      if nargin < 3 || isempty(movdata),
        movdata = struct;
      end

      obj.defaulttrkpat = defaulttrkpat;
      obj.defaulttrxpat = defaulttrxpat;
%      obj.defaultdetectpat = defaultdetectpat;
      obj.initMovData(movdata);

      obj.createGUI();      
    end
    
    function [movdata,dostore] = run(obj)
      uiwait(obj.gdata.fig);
      movdata = obj.movdata;
      dostore = obj.dostore;
    end
    
    function initMovData(obj,movdata)
      
      obj.movdata = movdata;

      if ~isfield(obj.movdata,'movfiles'),
        obj.movdata.movfiles = repmat({''},[1,obj.nview]);
      end
      if ~isfield(obj.movdata,'trkfiles'),
        obj.movdata.trkfiles = repmat({''},[1,obj.nview]);
      end
      if ~isempty(obj.defaulttrkpat)
        for ivw=1:obj.nview
          movI = obj.movdata.movfiles{ivw};
          if ~isempty(movI) && isempty(obj.movdata.trkfiles{ivw})
            obj.movdata.trkfiles{ivw} = obj.genTrkfile(movI,obj.defaulttrkpat);
          end
        end
      end
      % if obj.isma && ~isfield(obj.movdata,'detectfiles') 
      %   obj.movdata.detectfiles = repmat({''},[1,obj.nview]);
      % end
      % if ~isempty(obj.defaultdetectpat)
      %   for ivw=1:obj.nview
      %     movI = obj.movdata.movfiles{ivw};
      %     if ~isempty(movI) && ~isempty(obj.movdata.detectfiles) && isempty(obj.movdata.detectfiles{ivw})
      %       obj.movdata.detectfiles{ivw} = obj.genTrkfile(movI,obj.defaultdetectpat);
      %     end
      %   end
      % end
      if obj.hastrx && ~isfield(obj.movdata,'trxfiles'),
        obj.movdata.trxfiles = repmat({''},[1,obj.nview]);
      end
      if obj.hastrx && ~isempty(obj.defaulttrxpat)
        for ivw=1:obj.nview
          movI = obj.movdata.movfiles{ivw};
          if ~isempty(movI) && isempty(obj.movdata.trxfiles{ivw})
            obj.movdata.trxfiles{ivw} = obj.genTrkfile(movI,...
              obj.defaulttrxpat,'enforceExt',false);
          end
        end
      end      
      if obj.iscrop && ~isfield(obj.movdata,'cropRois'),
        obj.movdata.cropRois = repmat({nan(1,4)},[1,obj.nview]);
      end
      if ~isfield(obj.movdata,'calibrationfiles')
        obj.movdata.calibrationfiles = [];
      elseif iscell(obj.movdata.calibrationfiles)
        obj.movdata.calibrationfiles = cell2mat(obj.movdata.calibrationfiles);
      end
      if ~isfield(obj.movdata,'targets'),
        obj.movdata.targets = [];
      end
      if iscell(obj.movdata.targets),
        obj.movdata.targets = cell2mat(obj.movdata.targets);
      end
      if ~isfield(obj.movdata,'f0s'),
        obj.movdata.f0s = [];
      end
      if iscell(obj.movdata.f0s),
        obj.movdata.f0s = obj.movdata.f0s{1};
      end
      if ~isfield(obj.movdata,'f1s'),
        obj.movdata.f1s = [];
      end
      if iscell(obj.movdata.f1s),
        obj.movdata.f1s = obj.movdata.f1s{1};
      end
      if ~isfield(obj.movdata,'link_type'),
        obj.movdata.link_type = 'motion';
      end
      
      obj.rowinfo = struct;
      obj.rowinfo.movie = struct;
      obj.rowinfo.movie.prompt = 'Movie';
      obj.rowinfo.movie.movdatafield = 'movfiles';
      obj.rowinfo.movie.ext = {'*.avi;*.mp4;*.mjpg;*.ufmf','All video files (*.avi, *.mp4, *.mjpg, *.ufmf)'};
      obj.rowinfo.movie.type = 'inputfile';
      obj.rowinfo.movie.isvalperview = true;
      
      obj.rowinfo.trk = struct;
      obj.rowinfo.trk.prompt = 'Output trk file';
      obj.rowinfo.trk.movdatafield = 'trkfiles';
      obj.rowinfo.trk.ext = '*.trk';
      obj.rowinfo.trk.type = 'outputfile';
      obj.rowinfo.trk.isvalperview = true;
      
      % obj.rowinfo.detect = struct;
      % obj.rowinfo.detect.prompt = 'Output detection file';
      % obj.rowinfo.detect.movdatafield = 'detectfiles';
      % obj.rowinfo.detect.ext = '*.trk';
      % obj.rowinfo.detect.type = 'outputfile';
      % obj.rowinfo.detect.isvalperview = true;
      
      obj.rowinfo.trx = struct;
      obj.rowinfo.trx.prompt = 'Multitarget trx file';
      obj.rowinfo.trx.movdatafield = 'trxfiles';
      obj.rowinfo.trx.ext = '*.mat';
      obj.rowinfo.trx.type = 'inputfile';
      obj.rowinfo.trx.isvalperview = true;
      
      obj.rowinfo.cal = struct;
      obj.rowinfo.cal.prompt = 'Calibration file';
      obj.rowinfo.cal.movdatafield = 'calibrationfiles';
      obj.rowinfo.cal.ext = '*.*';
      obj.rowinfo.cal.type = 'inputfile';
      obj.rowinfo.cal.isvalperview = false;
      obj.rowinfo.cal.isoptional = ~obj.docalibrate;
      
      obj.rowinfo.crop = struct;
      obj.rowinfo.crop.prompt = 'Crop ROI (x1,x2,y1,y2)';
      obj.rowinfo.crop.movdatafield = 'cropRois';
      obj.rowinfo.crop.type = 'array';
      obj.rowinfo.crop.arraysize = [1,4];      
      obj.rowinfo.crop.isvalperview = true;
      % AL20201213. In some sense the crop is always optional, as one can
      % imagine the movie-to-be-trked to have arbitrary dims. That said,
      % in 99% of cases presumably the movie-to-be-trked will have the same
      % dims as the training movs. So we require crops if the proj has
      % crops. 
      obj.rowinfo.crop.isoptional = ~obj.lObj.cropProjHasCrops;
      
      obj.rowinfo.targets = struct;
      obj.rowinfo.targets.prompt = 'Targets (optional)';
      obj.rowinfo.targets.movdatafield = 'targets';
      obj.rowinfo.targets.type = 'array';
      obj.rowinfo.targets.isvalperview = false;
      obj.rowinfo.targets.isoptional = true;
      obj.rowinfo.targets.hasdetails = false;
      
      obj.rowinfo.f0s = struct;
      obj.rowinfo.f0s.prompt = 'Start frame (optional)';
      obj.rowinfo.f0s.movdatafield = 'f0s';
      obj.rowinfo.f0s.type = 'number';
      obj.rowinfo.f0s.isvalperview = false;
      obj.rowinfo.f0s.isoptional = true;
      obj.rowinfo.f0s.hasdetails = false;
      
      obj.rowinfo.f1s = struct;
      obj.rowinfo.f1s.prompt = 'End frame (optional)';
      obj.rowinfo.f1s.movdatafield = 'f1s';
      obj.rowinfo.f1s.type = 'number';
      obj.rowinfo.f1s.isvalperview = false;
      obj.rowinfo.f1s.isoptional = true;
      obj.rowinfo.f1s.hasdetails = false;
      
    end
    
    function createGUI(obj)
      
      % movies, trks, trx, crop, calibration
      obj.nfields = 2*obj.nview + double(obj.hastrx)*(obj.nview+1) + ... 
        double(obj.iscrop)*obj.nview + double(obj.nview>1) + ... 
        double(obj.isma)*obj.nview + 2;
      figname = 'Specify movie to track';
      
      obj.colorinfo.backgroundcolor = [0,0,0];
      obj.colorinfo.editfilecolor = [.2,.2,.2];
      obj.colorinfo.goodcolor = [1,1,1];
      obj.colorinfo.badcolor = [1,0,0];

      % compute the figure size
      units = get(obj.hParent,'units');
      set(obj.hParent,'Units','pixels');
      mainfigpos = get(obj.hParent,'Position');
      set(obj.hParent,'Units',units);
      figsz = [mainfigpos(3)*.8,20*(obj.nfields+2)+30]; % width, height
      mainfigctr = mainfigpos([1,2]) + mainfigpos([3,4])/2;
      obj.posinfo.figpos = [mainfigctr-figsz/2,figsz];
      % to do: make sure this is on the screen
      
      obj.gdata = struct;
      obj.gdata.fig = figure(...
        'menubar','none',...
        'toolbar','none',...
        'name',figname,...
        'NumberTitle','off',...
        'IntegerHandle','off',...
        'Tag','figure_SpecifyMovieToTrack',...
        'color',obj.colorinfo.backgroundcolor,...
        'units','pixels',...
        'position',obj.posinfo.figpos,...
        'ResizeFcn',@(src,evt) obj.figureResizeCallback(src,evt));
        
      set(obj.gdata.fig,'Units','normalized');
      obj.posinfo.figpos = get(obj.gdata.fig,'Position');

      obj.posinfo.border = .025;
      obj.posinfo.rowh = .05;
      obj.posinfo.rowborder = .01;
      obj.posinfo.colborder = .005;
      
      % border text border edit colborder filebutton border
      obj.posinfo.textx = obj.posinfo.border;
      obj.posinfo.textw = .3;
      obj.posinfo.detailsbuttonw = .05;
      obj.posinfo.editx = obj.posinfo.textx+obj.posinfo.textw+obj.posinfo.border;
      obj.posinfo.editw = 1-3*obj.posinfo.border-obj.posinfo.colborder-obj.posinfo.textw-obj.posinfo.detailsbuttonw;
      obj.posinfo.detailsbuttonx = 1-obj.posinfo.border-obj.posinfo.detailsbuttonw;
      obj.posinfo.controly = obj.posinfo.border;
      maxallrowh = 1-3*obj.posinfo.border-obj.posinfo.controly-(obj.nfields-1)*obj.posinfo.rowborder;
      obj.posinfo.rowh = maxallrowh/(obj.nfields+1);
      obj.posinfo.rowys = 1-obj.posinfo.border+obj.posinfo.rowborder - (obj.posinfo.rowborder+obj.posinfo.rowh)*(1:obj.nfields);
            
      % 1 - controly - 3*border - (n-1)*rowborder= (n+1)*rowh
      % border
      % rowh
      % rowborder
      % rowh
      % rowborder
      % rowh
      % border
      % rowh
      % border
      
      if obj.isma && obj.detailed_options
        controlbuttonstrs = {'Detect','Link','Track','Cancel'};
        controlbuttontags = {'detect','link','track','cancel'};
        controlbuttoncolors = ...
          [0,0,.8
          0,0,.8
          0,0,.8
          0,.7,.7];
      end
      controlbuttonstrs = {'Done','Cancel'};
      controlbuttontags = {'done','cancel'};
      controlbuttoncolors = ...
        [0,0,.8
        0,.7,.7];
      ncontrolbuttons = numel(controlbuttonstrs);
      controlbuttonw = .15;
      allcontrolbuttonw = ncontrolbuttons*controlbuttonw + (ncontrolbuttons-1)*obj.posinfo.colborder;
      controlbuttonx1 = .5-allcontrolbuttonw/2;
      controlbuttonxs = controlbuttonx1 + (controlbuttonw+obj.posinfo.colborder)*(0:ncontrolbuttons-1);
      controlbuttony = obj.posinfo.border;
      
      
      for i = 1:ncontrolbuttons,
        obj.gdata.button_control(i) = uicontrol('Style','pushbutton','String',controlbuttonstrs{i},...
          'ForegroundColor','w','BackgroundColor',controlbuttoncolors(i,:),'FontWeight','bold',...
          'Units','normalized','Enable','on','Position',[controlbuttonxs(i),controlbuttony,controlbuttonw,obj.posinfo.rowh],...
          'Tag',sprintf('controlbutton_%s',controlbuttontags{i}),...
          'Callback',@(h,e) obj.pb_control_Callback(h,e,controlbuttontags{i}));
      end

      % Add path display toggle buttons to the right
      pathToggleW = 0.08;
      pathToggleSpacing = 0.005;
      pathToggleX1 = 1 - obj.posinfo.border - 2*pathToggleW - pathToggleSpacing;

      % Load icon images
      leftAlignIcon = imread(fullfile(fileparts(mfilename('fullpath')), 'util', 'align_left.png'));
      rightAlignIcon = imread(fullfile(fileparts(mfilename('fullpath')), 'util', 'align_right.png'));
      if ndims(leftAlignIcon)==2
        leftAlignIcon = repmat(leftAlignIcon,[1 1 3]);
      end
      if ndims(rightAlignIcon)==2
        rightAlignIcon = repmat(rightAlignIcon,[1 1 3]);
      end

      % Create button group for path toggle buttons
      obj.gdata.bg_path = uibuttongroup(obj.gdata.fig,...
        'BackgroundColor',obj.colorinfo.backgroundcolor,...
        'BorderType','none',...
        'Units','normalized',...
        'Position',[pathToggleX1 controlbuttony 2*pathToggleW+pathToggleSpacing obj.posinfo.rowh],...
        'SelectionChangedFcn',@(src,evt) obj.pathToggleChanged(src,evt));

      % "Starts" toggle button (left side)
      obj.gdata.tb_path_starts = uitogglebutton(obj.gdata.bg_path,...
        'Text','',...
        'Icon',leftAlignIcon,...
        'Tooltip','Show path starts',...
        'FontColor','w','BackgroundColor',[1,1,1],...
        'FontWeight','bold',...
        'Value',~obj.showPathEnds,...
        'Tag','togglebutton_path_starts');

      % "Ends" toggle button (right side)
      obj.gdata.tb_path_ends = uitogglebutton(obj.gdata.bg_path,...
        'Text','',...
        'Icon',rightAlignIcon,...
        'Tooltip','Show path ends',...
        'FontColor','w','BackgroundColor',[1,1,1],...
        'FontWeight','bold',...
        'Value',obj.showPathEnds,...
        'Tag','togglebutton_path_ends');

      % Position the toggle buttons
      obj.updatePathTogglePositions();
      
      rowi = 1;
      for i = 1:obj.nview,
        tag = 'movie';
        if obj.nview > 1,
          str = sprintf('Movie view %d:',i);
        else
          str = 'Movie:';
        end
        obj.addRow(obj.posinfo.rowys(rowi),tag,i,str,obj.movdata.movfiles{i});
        rowi = rowi + 1;
      end
            
      for i = 1:obj.nview,
        tag = 'trk';
        if obj.nview > 1,
          str = sprintf('Output trk view %d:',i);
        else
          str = 'Output trk:';
        end
        obj.addRow(obj.posinfo.rowys(rowi),tag,i,str,obj.movdata.trkfiles{i});
        rowi = rowi + 1;
      end
      
      if obj.nview > 1,
        tag = 'cal';
        i = [];
        if obj.docalibrate,
          str = 'Calibration file:';
        else
          str = 'Calibration file (optional):';
        end
        obj.addRow(obj.posinfo.rowys(rowi),tag,i,str,obj.movdata.calibrationfiles);
        rowi = rowi + 1;
      end
      
      if obj.hastrx,
        key = 'trx';
        for i = 1:obj.nview,
          if obj.nview > 1,
            str = sprintf('Input trx view %d:',i);
          else
            str = 'Input trx file:';
          end
          obj.addRow(obj.posinfo.rowys(rowi),key,i,str,obj.movdata.trxfiles{i});
          rowi = rowi + 1;
        end
      end
      
      if obj.iscrop,
        key = 'crop';
        for i = 1:obj.nview,
          if obj.nview > 1,
            str = sprintf('%s view %d:',obj.rowinfo.crop.prompt,i);
          else
            str = sprintf('%s:',obj.rowinfo.crop.prompt);
          end
          if isempty(obj.movdata.cropRois),
            croproi = [];
          else
            croproi = obj.movdata.cropRois{i};
          end
          obj.addRow(obj.posinfo.rowys(rowi),key,i,str,croproi);
          rowi = rowi + 1;
        end
      end
      
      if obj.hastrx,
        key = 'targets';
        i = [];
        str = [obj.rowinfo.targets.prompt,':'];
        obj.addRow(obj.posinfo.rowys(rowi),key,i,str,obj.movdata.targets,...
          'List of target ids, e.g. 1 3 7',obj.rowinfo.targets.hasdetails);
        rowi = rowi + 1;
      end
      
      key = 'f0s';
      i = [];
      if ~isfield(obj.movdata,'f0s'),
        val = [];
      else
        val = obj.movdata.f0s;
      end
      str = [obj.rowinfo.f0s.prompt,':'];
      obj.addRow(obj.posinfo.rowys(rowi),key,i,str,val,...
        'Start of frame interval to track',obj.rowinfo.targets.hasdetails);
      rowi = rowi + 1;
      
      key = 'f1s';
      i = [];
      if ~isfield(obj.movdata,'f1s'),
        val = [];
      else
        val = obj.movdata.f1s;
      end
      str = [obj.rowinfo.f1s.prompt,':'];
      obj.addRow(obj.posinfo.rowys(rowi),key,i,str,val,...
        'End of frame interval to track',obj.rowinfo.targets.hasdetails);
      rowi = rowi + 1; %#ok<NASGU>

      if obj.isma && obj.detailed_options

        obj.gdata.linktext = ...
        uicontrol('Style','text','String','Linking Method',...
        'ForegroundColor','w','BackgroundColor',obj.colorinfo.backgroundcolor,'FontWeight','normal',...
        'Units','normalized','Position',[obj.posinfo.textx,obj.posinfo.rowys(rowi),obj.posinfo.textw,obj.posinfo.rowh],...
        'Tag','text_link',...
        'HorizontalAlignment','right',...
        'Parent',obj.gdata.fig);

        % Create button group for radio buttons
        obj.gdata.bg_linking = uibuttongroup(obj.gdata.fig,...
          'BackgroundColor',obj.colorinfo.backgroundcolor,...
          'BorderType','none',...
          'Position',[obj.posinfo.editx obj.posinfo.rowys(rowi) obj.posinfo.editw obj.posinfo.rowh],...
          'SelectionChangedFcn',@(src,evt) obj.linkingTypeChanged(src,evt));


        set(obj.gdata.bg_linking,'Units','pixels');
        pos = get(obj.gdata.bg_linking,'Position');
        set(obj.gdata.bg_linking,'Units','normalized');
        wd = pos(3)/2-20;  % Now divide by 2 instead of 3

        % Radio buttons - positions will be calculated dynamically
        obj.gdata.rb_motion = uiradiobutton(obj.gdata.bg_linking,...
          'Text','Motion Linking',...
          'FontColor','w',...
          'Position',[10 5 wd 20],...
          'Value',strcmp(obj.link_type,'motion'));

        obj.gdata.rb_identity = uiradiobutton(obj.gdata.bg_linking,...
          'Text','Identity linking',...
          'FontColor','w',...
          'Position',[wd+20 5 wd 20],...
          'Value',strcmp(obj.link_type,'identity'));
        
        % Update radio button positions after creation
        obj.updateLinkingButtonPositions();
        
        rowi = rowi + 1;

        % for i = 1:obj.nview,
      %     tag = 'detect';
      %     if obj.nview > 1,
      %       str = sprintf('Output detect view %d:',i);
      %     else
      %       str = 'Output detect:';
      %     end
      %     obj.addRow(obj.posinfo.rowys(rowi),tag,i,str,obj.movdata.detectfiles{i});
        % end

      end

      obj.updatePathDisplay();

      
    end
    
    function isgood = checkRowValue(obj,key,iview)
      
      isgood = false;
      ri = obj.rowinfo.(key);
      if isfield(ri,'isoptional'),
        isoptional = ri.isoptional;
      else
        isoptional = false;
      end
      val = obj.movdata.(ri.movdatafield);
      if ~isempty(iview),
        val = val{iview};
      end
      if strcmpi(ri.type,'inputfile'),
        if isoptional && isempty(val),
          isgood = true;
        else
          isgood = ~isempty(val) && exist(val,'file')>0;
        end
      elseif strcmpi(ri.type,'outputfile'),
        if ~isempty(val),
          p = fileparts(val);
          isgood = isempty(p) || (exist(p,'dir')>0);
        end
      elseif strcmpi(key,'crop'),
        if isempty(val),
          isgood = ~obj.lObj.cropProjHasCrops;
        else
          isgood = numel(val) == 4;
          if obj.lObj.cropProjHasCrops && isgood,
            isgood = ~any(isnan(val));
          end
        end
      elseif strcmpi(key,'targets'),
        isgood = all(val>0);
      elseif ismember(key,{'f0s','f1s'}),
        isgood = isempty(val) || ((numel(val) == 1) && (isnan(val) || (val > 0)));
      else
        warning('Unhandled type %s, setting isgood = true',key);
      end
      if isempty(iview),
        i = 1;
      else
        i = iview;
      end
      if isgood,
        color = obj.colorinfo.goodcolor;
      else
        color = obj.colorinfo.badcolor;
      end
      set(obj.gdata.(key).rowedit(i),'ForegroundColor',color);
      set(obj.gdata.(key).rowtext(i),'ForegroundColor',color);
      if isfield(obj.gdata.(key),'rowpb'),
        set(obj.gdata.(key).rowpb(i),'ForegroundColor',color);
      end
    end
    
    function addRow(obj,rowy,key,iview,textstr,editval,edittt,hasdetails)
      
      if ~exist('edittt','var'),
        edittt = '';
      end
      if ~exist('hasdetails','var'),
        hasdetails = true;
      end
      if isempty(iview),
        tag = key;
      else
        tag = sprintf('%s%d',key,iview);
      end
      i = iview;
      if isempty(i),
        i = 1;
      end
      obj.gdata.(key).rowtext(i) = ...
        uicontrol('Style','text','String',textstr,...
        'ForegroundColor','w','BackgroundColor',obj.colorinfo.backgroundcolor,'FontWeight','normal',...
        'Units','normalized','Position',[obj.posinfo.textx,rowy,obj.posinfo.textw,obj.posinfo.rowh],...
        'Tag',['text_',tag],...
        'HorizontalAlignment','right',...
        'Parent',obj.gdata.fig);
      if strcmpi(key,'crop') && all(isnan(editval)),
        editval = '';
      end
      if ismember(key,{'f0s','f1s'}) && all(isnan(editval)),
        editval = '';
      end
      if ischar(editval),
        editvalstr = editval;
      elseif strcmpi(key,'crop'),
        editvalstr = sprintf('%d  ',editval);
      else
        editvalstr = num2str(editval(:)');
      end
      cbk = @(hObject,eventdata)obj.rowedit_Callback(hObject,eventdata,key,iview);
      obj.gdata.(key).rowedit(i) = ...
        uicontrol('Style','edit','String',editvalstr,...
        'ForegroundColor','w','BackgroundColor',obj.colorinfo.editfilecolor,'FontWeight','normal',...
        'Units','normalized','Position',[obj.posinfo.editx,rowy,obj.posinfo.editw,obj.posinfo.rowh],...
        'Tag',['edit_',tag],...
        'ToolTip',edittt,...
        'HorizontalAlignment','left',...
        'Parent',obj.gdata.fig,...
        'Callback',cbk);
      if hasdetails,
        cbk = @(hObject,eventdata)obj.details_pb_Callback(hObject,eventdata,key,iview);
        obj.gdata.(key).rowpb(i) = ...
          uicontrol('Style','pushbutton','String','...',...
          'ForegroundColor','w','BackgroundColor',obj.colorinfo.editfilecolor,'FontWeight','normal',...
          'Units','normalized','Position',[obj.posinfo.detailsbuttonx,rowy,obj.posinfo.detailsbuttonw,obj.posinfo.rowh],...
          'Tag',['pb_',tag],...
          'HorizontalAlignment','center',...
          'Parent',obj.gdata.fig,...
          'Callback',cbk);
      end
      obj.isgood.(key)(i) = obj.checkRowValue(key,iview);
      
    end
    
    function rowedit_Callback(obj,hObject,eventdata,key,iview)
      
      ri = obj.rowinfo.(key);
      val = get(hObject,'String');
      if ismember(obj.rowinfo.(key).type,{'array'}),
        val = str2num(val); %#ok<ST2NM>
        if isempty(val),
          set(hObject,'String','');
          return;
        end
      elseif ismember(obj.rowinfo.(key).type,{'number'}),
        val = str2double(val);
        if isnan(val),
          set(hObject,'String','');
        end
      elseif ismember(obj.rowinfo.(key).type,{'inputfile','outputfile'}),
        file = java.io.File(val);
        if ~file.isAbsolute,
          val = char(file.getCanonicalPath());
          set(hObject,'String',val);
        end
      else
        error('Callback for %s not implemented',key);
      end
      
      if isempty(iview),
        obj.movdata.(ri.movdatafield) = val;
        i = 1;
      else
        obj.movdata.(ri.movdatafield){iview} = val;
        i = iview;
      end
      obj.isgood.(key)(i) = obj.checkRowValue(key,iview);
      
      if strcmp(ri.movdatafield,'movfiles')
        if ~isempty(obj.defaulttrkpat)
          movI = obj.movdata.movfiles{i};
          trkI = obj.genTrkfile(movI,obj.defaulttrkpat);
          obj.movdata.trkfiles{i} = trkI;
          obj.isgood.trk(i) = obj.checkRowValue('trk',i);
          set(obj.gdata.trk.rowedit(i),'String',trkI);
        end
        % if obj.isma && ~isempty(obj.defaultdetectpat)
        %   movI = obj.movdata.movfiles{i};
        %   trkI = obj.genTrkfile(movI,obj.defaultdetectpat);
        %   obj.movdata.detectfiles{i} = trkI;
        %   obj.isgood.detect(i) = obj.checkRowValue('detect',i);
        %   set(obj.gdata.detect.rowedit(i),'String',trkI);
        % end
        if obj.hastrx && ~isempty(obj.defaulttrxpat)
          movI = obj.movdata.movfiles{i};
          trxI = obj.genTrkfile(movI,obj.defaulttrxpat,'enforceExt',false);
          obj.movdata.trxfiles{i} = trxI;
          obj.isgood.trx(i) = obj.checkRowValue('trx',i);
          set(obj.gdata.trx.rowedit(i),'String',trxI);
        end
      end
      
    end

    
    function details_pb_Callback(obj,hObject,eventdata,key,iview)
      if ismember(obj.rowinfo.(key).type,{'inputfile','outputfile'}),
        obj.details_pb_file_Callback(hObject,eventdata,key,iview);
      elseif strcmpi(key,'crop'),
        obj.details_pb_crop_Callback(hObject,eventdata,key,iview);
      else
        error('callback for %s not implemented',key);
      end
    end

    function details_pb_file_Callback(obj,hObject,eventdata,key,iview) %#ok<INUSL>
      persistent lastpath;
      if isempty(lastpath),
        lastpath = '';
      end
      
      %fprintf('Callback: %s\n',key);
      
      if isempty(iview),
        i = 1;
      else
        i = iview;
      end

      ri = obj.rowinfo.(key);
      strti = ri.prompt;
      if obj.nview > 1 && ~isempty(iview),
        strti = sprintf('%s view %d',strti,iview);
      end
      defaultval = obj.movdata.(ri.movdatafield);
      if ~isempty(iview),
        defaultval = defaultval{iview};
      end
      if isempty(defaultval),
        defaultval = lastpath;
      end
      filterspec = ri.ext;
      isinput = strcmpi(ri.type,'inputfile');
      if isinput,
        [filename,pathname] = uigetfile(filterspec,strti,defaultval);
      else
        [filename,pathname] = uiputfile(filterspec,strti,defaultval);
      end
      if ~ischar(filename),
        return;
      end
      file = fullfile(pathname,filename);
      ex = exist(file,'file');
      if isinput && ~ex,
        errdlg(sprintf('File %s does not exist',file));
        return;
      end
      if isempty(iview),
        obj.movdata.(ri.movdatafield) = file;
      else
        obj.movdata.(ri.movdatafield){iview} = file;
      end
      
      set(obj.gdata.(key).rowedit(i),'String',file);
      lastpath = pathname;
      obj.isgood.(key)(i) = obj.checkRowValue(key,iview);      
      
      if strcmp(ri.movdatafield,'movfiles') && ~isempty(obj.defaulttrkpat)
        movI = obj.movdata.movfiles{i};
        trkI = obj.genTrkfile(movI,obj.defaulttrkpat);
        obj.movdata.trkfiles{i} = trkI;
        obj.isgood.trk(i) = obj.checkRowValue('trk',i);
        set(obj.gdata.trk.rowedit(i),'String',trkI);
      end
      if strcmp(ri.movdatafield,'movfiles') && ~isempty(obj.defaultdetectpat) && isfield(obj.gdata,'detect'),
        movI = obj.movdata.movfiles{i};
        trkI = obj.genTrkfile(movI,obj.defaultdetectpat);
        obj.movdata.detectfiles{i} = trkI;
        obj.isgood.detect(i) = obj.checkRowValue('detect',i);
        set(obj.gdata.detect.rowedit(i),'String',trkI);
      end
      if strcmp(ri.movdatafield,'movfiles') && obj.hastrx && ~isempty(obj.defaulttrxpat)
        movI = obj.movdata.movfiles{i};
        trxI = obj.genTrkfile(movI,obj.defaulttrxpat,'enforceExt',false);
        obj.movdata.trxfiles{i} = trxI;
        obj.isgood.trx(i) = obj.checkRowValue('trx',i);
        set(obj.gdata.trx.rowedit(i),'String',trxI);
      end
      
    end
    
    function details_pb_crop_Callback(obj,hObject,eventdata,key,iview) %#ok<INUSL>      
      %fprintf('Callback: %s\n',key);
      
      if ~obj.isgood.movie(iview),
        if obj.nview > 1,
          s = sprintf('for view %d ',iview);
        else
          s = '';
        end
        uiwait(msgbox(sprintf('Movie %smust be specified to use GUI to set crop region.',s)));
        return;
      end
      
      % movie reader
      if isempty(obj.movieReader) || numel(obj.movieReader) < iview || ...
          isempty(obj.movieReader{iview}),
        obj.movieReader{iview} = MovieReader;
        obj.movieReader{iview}.flipVert = obj.lObj.movieInvert(iview);
        obj.movieReader{iview}.forceGrayscale = obj.lObj.movieForceGrayscale;
        obj.movieReader{iview}.open(obj.movdata.movfiles{iview});
      end
      
      cropRoi = obj.movdata.cropRois{iview};
      if isempty(cropRoi) || all(isnan(cropRoi)),
        if isempty(obj.cropwh),
          cropRoi = [ceil(obj.movieReader{iview}.nc/4),ceil(3*obj.movieReader{iview}.nc/4),...
            ceil(obj.movieReader{iview}.nr/4),ceil(3*obj.movieReader{iview}.nr/4)];
        else
          x1 = max(1,round(obj.movieReader{iview}.nc/2-obj.cropwh(iview,1)/2));
          y1 = max(1,round(obj.movieReader{iview}.nr/2-obj.cropwh(iview,2)/2));
          x2 = x1 + obj.cropwh(iview,1) - 1;
          y2 = y1 + obj.cropwh(iview,2) - 1;
          cropRoi = [x1,x2,y1,y2];
        end
      end
      
      fr = ceil(obj.movieReader{iview}.nframes/2);
      im = obj.movieReader{iview}.readframe(fr);      
      obj.gdata.crop.fig = figure(...
        'name','Specify crop region by dragging box',...
        'NumberTitle','off',...
        'IntegerHandle','off',...
        'Tag','figure_SpecifyMovieToTrackCrop',...
        'color',obj.colorinfo.backgroundcolor,...
        'units','normalized');
      hax = axes('Position',[.05,.15,.9,.8],'Parent',obj.gdata.crop.fig);
      him = imagesc(im);
      axis(hax,'image');
      colormap(hax,'gray');
      set(hax,'XColor','w','YColor','w');
      
      controlbuttonstrs = {'Done','Cancel'};
      dosave = [true,false];
      controlbuttoncolors = ...
        [0,0,.8
        0,.7,.7];
      ncontrolbuttons = numel(controlbuttonstrs);
      controlbuttonw = .15;
      allcontrolbuttonw = ncontrolbuttons*controlbuttonw + (ncontrolbuttons-1)*obj.posinfo.colborder;
      controlbuttonx1 = .5-allcontrolbuttonw/2;
      controlbuttonxs = controlbuttonx1 + (controlbuttonw+obj.posinfo.colborder)*(0:ncontrolbuttons-1);
      controlbuttony = .05;
      controlbuttonh = .05;

      for i = 1:ncontrolbuttons,
        uicontrol('style','pushbutton','parent',obj.gdata.crop.fig,...
          'String',controlbuttonstrs{i},'Units','normalized',...
          'Position',[controlbuttonxs(i),controlbuttony,controlbuttonw,controlbuttonh],...
          'backgroundcolor',controlbuttoncolors(i,:),'foregroundcolor','w',...
          'fontweight','bold','Callback',@(h,e) obj.pb_crop_Callback(h,e,iview,dosave(i)));
      end
      
      if isempty(obj.cropwh),
        interactionsallowed = 'all';
      else
        interactionsallowed = 'translate';
      end
      pos = [cropRoi(1),cropRoi(3),cropRoi(2)-cropRoi(1)+1,cropRoi(4)-cropRoi(3)+1];
      obj.gdata.crop.rect = images.roi.Rectangle(hax,'Position',pos,...
        'InteractionsAllowed',interactionsallowed,...
        'Deletable',false);

      uiwait(obj.gdata.crop.fig);
      
    end
    
    function pb_control_Callback(obj,h,e,tag)

      if ismember(lower(tag),{'done','apply'}),

        fns = fieldnames(obj.isgood);
        isgood = true;
        for i = 1:numel(fns)
          isgood = isgood && all(obj.isgood.(fns{i}));
        end

        if ~isgood,
          uiwait(errordlg('Fields marked in red have missing or incorrect values. Correct them or press the Cancel button'));
          return;
        end

        % Check for existing detect files and prompt user
        userChoice = obj.checkAndPromptForDetectFiles();
        switch userChoice
          case 'cancel'
            return; % User cancelled, don't close dialog
          case 'use_detect'
            obj.movdata.docontinue = true;
          case 'new_tracking'
            obj.movdata.docontinue = false;
          case 'no_detect_files'
            obj.movdata.docontinue = false;
        end

        % Check for existing output trk files and ask user if they want to overwrite
        overwriteChoice = obj.checkAndPromptForOutputFiles();
        if strcmp(overwriteChoice, 'cancel')
          return; % User cancelled, don't close dialog
        end

        obj.dostore = true;
      end
      % if obj.isma
      %   if ismember(lower(tag),{'track','link','detect'})
      %     obj.link_type = lower(tag);
      %     obj.movdata.link_type = lower(tag);
      %   end
      % end
      if ismember(lower(tag),{'done','cancel'}),
        delete(obj.gdata.fig);
      end

    end
    
    function pb_crop_Callback(obj,h,e,iview,dosave)
      if ~isfield(obj.gdata.crop,'fig'),
        return;
      end
      if isfield(obj.gdata.crop,'rect'),
        if dosave,
          pos = obj.gdata.crop.rect.Position;
          x1 = round(pos(1));
          y1 = round(pos(2));
          if isempty(obj.cropwh),
            x2 = round(x1 + pos(3) - 1);
            y2 = round(y1 + pos(4) - 1);
          else
            x2 = x1 + obj.cropwh(1)-1;
            y2 = y1 + obj.cropwh(2)-1;
          end
          
          obj.setCropRoi(iview,[x1,x2,y1,y2],h);
        end
        delete(obj.gdata.crop.rect);
        obj.gdata.crop.rect = [];
      end
      if isfield(obj.gdata.crop,'fig'),
        hfig = obj.gdata.crop.fig;
        obj.gdata.crop.fig = [];
        delete(hfig);
      end
    end
    
    function setCropRoi(obj,iview,pos,h)
      if ~exist('h','var'),
        h = nan;
      end
      obj.movdata.cropRois{iview} = pos;
      if h ~= obj.gdata.crop.rowedit(iview),
        obj.gdata.crop.rowedit(iview).String = sprintf('%d  ',pos);
        obj.checkRowValue('crop',iview);
      end
    end

    function linkingTypeChanged(obj, src, evt)
      % Callback for linking type radio button group
      selectedButton = evt.NewValue;

      % Determine which linking type was selected
      if selectedButton == obj.gdata.rb_motion
        obj.link_type = 'motion';
      elseif selectedButton == obj.gdata.rb_identity
        obj.link_type = 'identity';
      end

      % Update toTrack data structure
      obj.movdata.link_type = obj.link_type;
    end

    function trk = genTrkfile(obj,movie,defaulttrk,varargin)
      if isempty(movie)
        trk = '';
      else
        trk = Labeler.genTrkFileName(defaulttrk,...
          obj.lObj.baseTrkFileMacros(),movie,varargin{:});
      end
    end
    
    function figureResizeCallback(obj, src, evt)
      % Called when the figure is resized - update radio button positions
      if obj.lObj.maIsMA && isfield(obj.gdata, 'bg_linking') && isvalid(obj.gdata.bg_linking)
        obj.updateLinkingButtonPositions();
      end
      % Update path toggle button positions on resize
      if isfield(obj.gdata, 'bg_path') && isvalid(obj.gdata.bg_path)
        obj.updatePathTogglePositions();
      end
      % Update path display after resize to adjust truncation to new component sizes
      obj.updatePathDisplay();
    end
    
    function updateLinkingButtonPositions(obj)
      % Calculate and update radio button positions based on button group size
      if ~isfield(obj.gdata, 'bg_linking') || ~isvalid(obj.gdata.bg_linking)
        return;
      end
      
      try
        % Update the "Linking Method" text position if it exists
        if isfield(obj.gdata, 'linktext') && isvalid(obj.gdata.linktext)
          % Get current button group position to align text with it
          bgPos_norm = get(obj.gdata.bg_linking,'Position');
          % Update text position to align with button group
          set(obj.gdata.linktext, 'Position', [obj.posinfo.textx, bgPos_norm(2), obj.posinfo.textw, obj.posinfo.rowh]);
        end
        
        % Get the button group position in pixels
        set(obj.gdata.bg_linking,'Units','pixels');
        bgPos = get(obj.gdata.bg_linking,'Position');
        bgWidth = bgPos(3);
        bgHeight = bgPos(4);
        set(obj.gdata.bg_linking,'Units','normalized');
        
        % Calculate button dimensions with padding
        padding = 10;  % pixels
        buttonHeight = max(20, bgHeight - 2*padding);  % minimum 20px height
        buttonWidth = (bgWidth - 3*padding) / 2;  % divide width by 2 with padding

        % Calculate positions for each button
        y = (bgHeight - buttonHeight) / 2;  % center vertically

        % Update positions (now only motion and identity)
        if isfield(obj.gdata, 'rb_motion') && isvalid(obj.gdata.rb_motion)
          obj.gdata.rb_motion.Position = [padding, y, buttonWidth, buttonHeight];
        end
        if isfield(obj.gdata, 'rb_identity') && isvalid(obj.gdata.rb_identity)
          obj.gdata.rb_identity.Position = [padding + buttonWidth + padding, y, buttonWidth, buttonHeight];
        end
        
      catch ME
        % Silently handle errors during resize
        warning(ME.identifier,'Error updating linking button positions: %s', ME.message);
      end
    end

    function pathToggleChanged(obj, src, evt)
      % Callback for path display toggle button group
      selectedButton = evt.NewValue;

      % Determine which toggle was selected and update showPathEnds accordingly
      if selectedButton == obj.gdata.tb_path_starts
        obj.showPathEnds = false;  % Show path starts
      elseif selectedButton == obj.gdata.tb_path_ends
        obj.showPathEnds = true;   % Show path ends
      end

      % Update all displayed file paths
      obj.updatePathDisplay();
    end

    function updatePathTogglePositions(obj)
      % Update positions of the path toggle buttons
      if ~isfield(obj.gdata, 'bg_path') || ~isvalid(obj.gdata.bg_path)
        return;
      end

      try
        % Get the button group position in pixels
        set(obj.gdata.bg_path,'Units','pixels');
        bgPos = get(obj.gdata.bg_path,'Position');
        bgWidth = bgPos(3);
        bgHeight = bgPos(4);
        set(obj.gdata.bg_path,'Units','normalized');

        % Calculate button dimensions - split the width in half
        buttonWidth = bgWidth / 2;
        buttonHeight = max(20, bgHeight - 4);  % Leave small padding

        % Position buttons side by side
        if isfield(obj.gdata, 'tb_path_starts') && isvalid(obj.gdata.tb_path_starts)
          obj.gdata.tb_path_starts.Position = [2, 2, buttonWidth-4, buttonHeight];
        end
        if isfield(obj.gdata, 'tb_path_ends') && isvalid(obj.gdata.tb_path_ends)
          obj.gdata.tb_path_ends.Position = [buttonWidth+2, 2, buttonWidth-4, buttonHeight];
        end

      catch ME
        % Silently handle errors during positioning
        warning(ME.identifier,'Error updating path toggle positions: %s', ME.message);
      end
    end

    function updatePathDisplay(obj)
      % Update the display of file paths based on current mode (starts vs ends)

      % Update movie file paths
      if isfield(obj.gdata, 'movie') && isfield(obj.gdata.movie, 'rowedit')
        for i = 1:obj.nview
          if i <= length(obj.gdata.movie.rowedit) && isvalid(obj.gdata.movie.rowedit(i))
            originalPath = obj.movdata.movfiles{i};
            if ~isempty(originalPath)
              displayPath = obj.getDisplayPath(originalPath, obj.gdata.movie.rowedit(i));
              set(obj.gdata.movie.rowedit(i), 'String', displayPath);
            end
          end
        end
      end

      % Update trk file paths
      if isfield(obj.gdata, 'trk') && isfield(obj.gdata.trk, 'rowedit')
        for i = 1:obj.nview
          if i <= length(obj.gdata.trk.rowedit) && isvalid(obj.gdata.trk.rowedit(i))
            originalPath = obj.movdata.trkfiles{i};
            if ~isempty(originalPath)
              displayPath = obj.getDisplayPath(originalPath, obj.gdata.trk.rowedit(i));
              set(obj.gdata.trk.rowedit(i), 'String', displayPath);
            end
          end
        end
      end

      % Update trx file paths (if project has trx)
      if obj.hastrx && isfield(obj.gdata, 'trx') && isfield(obj.gdata.trx, 'rowedit')
        for i = 1:obj.nview
          if i <= length(obj.gdata.trx.rowedit) && isvalid(obj.gdata.trx.rowedit(i))
            originalPath = obj.movdata.trxfiles{i};
            if ~isempty(originalPath)
              displayPath = obj.getDisplayPath(originalPath, obj.gdata.trx.rowedit(i));
              set(obj.gdata.trx.rowedit(i), 'String', displayPath);
            end
          end
        end
      end

      % Update calibration file path (if single path for all views)
      if isfield(obj.gdata, 'cal') && isfield(obj.gdata.cal, 'rowedit') && ...
         length(obj.gdata.cal.rowedit) >= 1 && isvalid(obj.gdata.cal.rowedit(1))
        originalPath = obj.movdata.calibrationfiles;
        if ~isempty(originalPath)
          displayPath = obj.getDisplayPath(originalPath, obj.gdata.cal.rowedit(1));
          set(obj.gdata.cal.rowedit(1), 'String', displayPath);
        end
      end
    end

    function displayPath = getDisplayPath(obj, originalPath, editComponent)
      % Get the appropriate display path based on current toggle setting
      if isempty(originalPath)
        displayPath = '';
        return;
      end

      if obj.showPathEnds
        % Show truncated path ends
        displayPath = PathTruncationUtils.truncateFilePath(...
          originalPath, 'component', editComponent, 'startFraction', 0);
      else
        % Show whole path when showing path starts
        displayPath = originalPath;
      end
    end

    function userChoice = checkAndPromptForDetectFiles(obj)
      % Check for existing _detect files and prompt user if they want to continue with them
      % Returns user choice: 'cancel', 'use_detect', 'new_tracking', or 'no_detect_files'

      % Get all trk files from the current movie data
      trkfiles = {};
      if isfield(obj.movdata, 'trkfiles') && ~isempty(obj.movdata.trkfiles)
        trkfiles = obj.movdata.trkfiles;
      elseif isfield(obj.movdata, 'trkfile') && ~isempty(obj.movdata.trkfile)
        trkfiles = {obj.movdata.trkfile};
      end

      if isempty(trkfiles)
        userChoice = 'no_detect_files';
        return;
      end

      existingDetectFiles = {};
      detectTimestamps = {};
      predictionStatus = {};

      for i = 1:numel(trkfiles)
        if isempty(trkfiles{i})
          continue;
        end

        % Generate corresponding _detect filename and part filename
        [pathpart, namepart, ext] = fileparts(trkfiles{i});
        detectFile = fullfile(pathpart, [namepart '_detect' ext]);
        detectPartFile = fullfile(pathpart, [namepart '_detect' ext '.part']);

        % Check if _detect file or _detect.part file exists
        detectExists = exist(detectFile, 'file');
        detectPartExists = exist(detectPartFile, 'file');

        if detectExists || detectPartExists
          % Use detect file if it exists, otherwise use part file
          if detectExists
            fileToUse = detectFile;
            status = 'Full';
            fileInfo = dir(detectFile);
          else
            fileToUse = detectPartFile;
            status = 'Partial';
            fileInfo = dir(detectPartFile);
          end

          existingDetectFiles{end+1} = fileToUse; %#ok<AGROW>
          predictionStatus{end+1} = status; %#ok<AGROW>

          % Get file timestamp
          if ~isempty(fileInfo)
            detectTimestamps{end+1} = datestr(fileInfo.datenum, 'yyyy-mm-dd HH:MM:SS'); %#ok<AGROW>
          else
            detectTimestamps{end+1} = 'Unknown'; %#ok<AGROW>
          end
        end
      end

      if isempty(existingDetectFiles)
        userChoice = 'no_detect_files';
        return;
      end

      % Create prompt message
      promptMsg = sprintf('Found %d existing detection file(s):\n\n', numel(existingDetectFiles));
      for i = 1:numel(existingDetectFiles)
        [~, fname, fext] = fileparts(existingDetectFiles{i});
        promptMsg = sprintf('%s%s%s (%s) - %s\n', promptMsg, fname, fext, predictionStatus{i}, detectTimestamps{i});
      end
      promptMsg = sprintf('%s\nDo you want to use these existing detection files or start new tracking?', promptMsg);

      % Show dialog
      choice = questdlg(promptMsg, 'Existing Detection Files Found', ...
        'Use Existing', 'New Tracking', 'Cancel', 'Use Existing');

      switch choice
        case 'Use Existing'
          userChoice = 'use_detect';
        case 'New Tracking'
          userChoice = 'new_tracking';
        case 'Cancel'
          userChoice = 'cancel';
        otherwise
          userChoice = 'cancel';
      end
    end

    function userChoice = checkAndPromptForOutputFiles(obj)
      % Check for existing output trk files and prompt user if they want to overwrite them
      % Returns user choice: 'cancel', 'overwrite', or 'no_output_files'

      % Get all trk files that will be created as output
      trkfiles = {};
      if isfield(obj.movdata, 'trkfiles') && ~isempty(obj.movdata.trkfiles)
        trkfiles = obj.movdata.trkfiles;
      elseif isfield(obj.movdata, 'trkfile') && ~isempty(obj.movdata.trkfile)
        trkfiles = {obj.movdata.trkfile};
      end

      if isempty(trkfiles)
        userChoice = 'no_output_files';
        return;
      end

      existingOutputFiles = {};
      outputTimestamps = {};

      for i = 1:numel(trkfiles)
        if isempty(trkfiles{i})
          continue;
        end

        % Check if output trk file already exists
        if exist(trkfiles{i}, 'file')
          existingOutputFiles{end+1} = trkfiles{i}; %#ok<AGROW>

          % Get file timestamp
          fileInfo = dir(trkfiles{i});
          if ~isempty(fileInfo)
            outputTimestamps{end+1} = datestr(fileInfo.datenum, 'yyyy-mm-dd HH:MM:SS'); %#ok<AGROW>
          else
            outputTimestamps{end+1} = 'Unknown'; %#ok<AGROW>
          end
        end
      end

      if isempty(existingOutputFiles)
        userChoice = 'no_output_files';
        return;
      end

      % Create prompt message
      promptMsg = sprintf('Found %d existing output tracking file(s):\n\n', numel(existingOutputFiles));
      for i = 1:numel(existingOutputFiles)
        [~, fname, fext] = fileparts(existingOutputFiles{i});
        promptMsg = sprintf('%s%s%s - %s\n', promptMsg, fname, fext, outputTimestamps{i});
      end
      promptMsg = sprintf('%s\nDo you want to overwrite these existing tracking files?', promptMsg);

      % Show dialog
      choice = questdlg(promptMsg, 'Existing Output Files Found', ...
        'Overwrite', 'Cancel', 'Overwrite');

      switch choice
        case 'Overwrite'
          userChoice = 'overwrite';
        case 'Cancel'
          userChoice = 'cancel';
        otherwise
          userChoice = 'cancel';
      end
    end

  end
end
