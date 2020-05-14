classdef SpecifyMovieToTrackGUI < handle
  
  properties 
    defaultdir = '';
    lObj = [];
    hParent = [];
    movdata = [];
    nview = 1;
    hastrx = false;
    iscrop = false;
    docalibrate = false;
    nfields = nan;
    gdata = [];
    posinfo = struct;
    colorinfo = struct;
    rowinfo = struct;
    isgood = struct;
    cropwh = [];
    movieReader = [];
  end
  methods
    function obj = SpecifyMovieToTrackGUI(lObj,hParent,movdata)

      obj.lObj = lObj;
      obj.hParent = hParent;

      obj.nview = obj.lObj.nview;
      obj.hastrx = obj.lObj.hasTrx;
      obj.iscrop = true;
      %obj.iscrop = obj.lObj.cropIsCropMode; % to do: allow cropping when trained without cropping?
      obj.docalibrate = obj.nview > 1 && (obj.lObj.trackParams.ROOT.PostProcess.reconcile3dType);
      if obj.iscrop,
        if obj.lObj.cropIsCropMode,
          obj.cropwh = obj.lObj.cropGetCurrentCropWidthHeightOrDefault();
        else
          obj.cropwh = [];
        end
      end

      if nargin < 3 || isempty(movdata),
        movdata = struct;
      end

      obj.initMovData(movdata);

      obj.createGUI();
      
    end
    
    function movdata = run(obj)
      uiwait(obj.gdata.fig);
      movdata = obj.movdata;
    end
    
    function initMovData(obj,movdata)
      
      obj.movdata = movdata;

      if ~isfield(obj.movdata,'movfiles'),
        obj.movdata.movfiles = repmat({''},[1,obj.nview]);
      end
      if ~isfield(obj.movdata,'trkfiles'),
        obj.movdata.trkfiles = repmat({''},[1,obj.nview]);
      end
      if obj.hastrx && ~isfield(obj.movdata,'trxfiles'),
        obj.movdata.trxfiles = repmat({''},[1,obj.nview]);
      end
      if obj.iscrop && ~isfield(obj.movdata,'cropRois'),
        obj.movdata.cropRois = repmat({nan(1,4)},[1,obj.nview]);
      end
      if obj.docalibrate && ~isfield(obj.movdata,'calibrationfiles'),
        obj.movdata.calibrationfiles = '';
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
      
      obj.rowinfo.crop = struct;
      obj.rowinfo.crop.prompt = 'Crop ROI (x,y,w,h)';
      obj.rowinfo.crop.movdatafield = 'cropRois';
      obj.rowinfo.crop.type = 'array';
      obj.rowinfo.crop.arraysize = [1,4];      
      obj.rowinfo.crop.isvalperview = true;
      
    end
    
    function createGUI(obj)
      
      % movies, trks, trx, crop, calibration
      obj.nfields = 2*obj.nview + double(obj.hastrx)*obj.nview + double(obj.iscrop)*obj.nview + double(obj.nview>1);
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
        'position',obj.posinfo.figpos);
        
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
        if docalibration
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
            str = sprintf('Crop box (x,y,w,h) view %d:',i);
          else
            str = 'Crop box (x,y,w,h):';
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
           
    end
    
    function isgood = checkRowValue(obj,key,iview)
      
      isgood = false;
      ri = obj.rowinfo.(key);
      if strcmpi(ri.type,'inputfile'),
        val = obj.movdata.(ri.movdatafield);
        if ~isempty(iview),
          val = val{iview};
        end
        isgood = exist(val,'file')>0;
      elseif strcmpi(ri.type,'outputfile'),
        val = obj.movdata.(ri.movdatafield);
        if ~isempty(iview),
          val = val{iview};
        end
        if ~isempty(val),
          p = fileparts(val);
          isgood = isempty(p) || (exist(p,'dir')>0);
        end
      elseif strcmpi(key,'crop'),
        if isempty(obj.movdata.(ri.movdatafield)),
          isgood = ~obj.lObj.cropIsCropMode;
        else
          val = obj.movdata.(ri.movdatafield){iview};
          isgood = numel(val) == 4;
          if obj.lObj.cropIsCropMode && isgood,
            isgood = ~any(isnan(val));
          end
        end
      else
        warning('Unhandled type %s, setting isgood = true',ri.key);
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
      set(obj.gdata.(key).rowpb(i),'ForegroundColor',color);
    end
    
    function addRow(obj,rowy,key,iview,textstr,editval,edittt)
      
      if ~exist('edittt','var'),
        edittt = '';
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
      if ischar(editval),
        editvalstr = editval;
      else
        editvalstr = num2str(editval);
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
      cbk = @(hObject,eventdata)obj.details_pb_Callback(hObject,eventdata,key,iview);
      obj.gdata.(key).rowpb(i) = ...
        uicontrol('Style','pushbutton','String','...',...
        'ForegroundColor','w','BackgroundColor',obj.colorinfo.editfilecolor,'FontWeight','normal',...
        'Units','normalized','Position',[obj.posinfo.detailsbuttonx,rowy,obj.posinfo.detailsbuttonw,obj.posinfo.rowh],...
        'Tag',['pb_',tag],...
        'HorizontalAlignment','center',...
        'Parent',obj.gdata.fig,...
        'Callback',cbk);
      obj.isgood.(key)(i) = obj.checkRowValue(key,iview);
      
    end
    
    function rowedit_Callback(obj,hObject,eventdata,key,iview)
      
      ri = obj.rowinfo.(key);
      val = get(hObject,'String');
      if ismember(obj.rowinfo.(key).type,{'array'}),
        val = str2num(val); %#ok<ST2NM>
      elseif ismember(obj.rowinfo.(key).type,{'inputfile','outputfile'}),
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
      
      fprintf('Callback: %s\n',key);
      
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
      
    end
    
    function details_pb_crop_Callback(obj,hObject,eventdata,key,iview) %#ok<INUSL>      
      fprintf('Callback: %s\n',key);
      
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
      if isempty(obj.movieReader),
        obj.movieReader = MovieReader;
        obj.movieReader.flipVert = obj.lObj.movieInvert(iview);
        obj.movieReader.forceGrayscale = obj.lObj.movieForceGrayscale;
        obj.movieReader.open(obj.movdata.movfiles{iview});
      end
      
      cropRoi = obj.movdata.cropRois{iview};
      if isempty(cropRoi) || all(isnan(cropRoi)),
        if isempty(obj.cropwh),
          cropRoi = [1,1,obj.movieReader.nc,obj.movieReader.nr];
        else
          x = max(1,obj.movieReader.nc/2-obj.cropwh(iview,1)/2);
          y = max(1,obj.movieReader.nr/2-obj.cropwh(iview,2)/2);
          cropRoi = [x,y,obj.cropwh(iview,:)];
        end
      end
      
      fr = ceil(obj.movieReader.nframes/2);
      im = obj.movieReader.readframe(fr);      
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
      obj.gdata.crop.rect = images.roi.Rectangle(hax,'Position',cropRoi,...
        'InteractionsAllowed',interactionsallowed,...
        'Deletable',false);

      uiwait(obj.gdata.crop.fig);
      
%       if isempty(iview),
%         i = 1;
%       else
%         i = iview;
%       end
% 
%       ri = obj.rowinfo.(key);
%       strti = ri.prompt;
%       if obj.nview > 1 && ~isempty(iview),
%         strti = sprintf('%s view %d',strti,iview);
%       end
%       defaultval = obj.movdata.(ri.movdatafield);
%       if ~isempty(iview),
%         defaultval = defaultval{iview};
%       end
%       if isempty(defaultval),
%         defaultval = lastpath;
%       end
%       filterspec = ri.ext;
%       isinput = strcmpi(ri.type,'inputfile');
%       if isinput,
%         [filename,pathname] = uigetfile(filterspec,strti,defaultval);
%       else
%         [filename,pathname] = uiputfile(filterspec,strti,defaultval);
%       end
%       if ~ischar(filename),
%         return;
%       end
%       file = fullfile(pathname,filename);
%       ex = exist(file,'file');
%       if isinput && ~ex,
%         errdlg(sprintf('File %s does not exist',file));
%         return;
%       end
%       if isempty(iview),
%         obj.movdata.(ri.movdatafield) = file;
%       else
%         obj.movdata.(ri.movdatafield){iview} = file;
%       end
%       
%       set(obj.gdata.rowedit.(key)(i),'String',file);
%       lastpath = pathname;
%       obj.isgood.(key)(i) = obj.checkRowValue(key,iview);
%       
    end
    
    function pb_control_Callback(obj,h,e,tag)
      
      if ismember(lower(tag),{'done','apply'}),
        
        fns = fieldnames(obj.isgood);
        isgood = true;
        for i = 1:numel(fns),
          isgood = isgood && all(obj.isgood.(fns{i}));
        end
        
        if ~isgood,
          uiwait(errordlg('Fields marked in red have missing or incorrect values. Correct them or press the Cancel button'));
          return;
        end
      end
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
          obj.setCropRoi(iview,pos,h);
          obj.movdata.cropRois{iview} = pos;
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
        obj.gdata.crop.rowedit(iview).String = sprintf('%.1f  ',pos);
        obj.checkRowValue('crop',iview);
      end
    end
    
  end
end
