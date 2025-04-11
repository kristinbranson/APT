classdef TrackBatchGUI < handle

  properties
    toTrack = [];
    lObj = [];
    hParent = [];
    isma = [];
    nmovies = 0;
    page = 1;
    npages = 1;
    posinfo = struct;
    gdata = struct;
    nmovies_per_page = 10;
    isbusy = false;
    
    needsSave = false; % if true, .toTrack has changes relative to last save/loaded json
    
    defaulttrkpat;
    defaulttrxpat = '$movdir/trx.mat';
    defaultdetectpat;
  end
  
  methods
    function obj = TrackBatchGUI(lObj,varargin)
      obj.lObj = lObj;
      obj.isma = lObj.maIsMA;
      obj.hParent = obj.lObj.gdata.mainFigure_;      
      toTrack = myparse(varargin,'toTrack',struct);
      
      obj.defaulttrkpat = lObj.defaultExportTrkRawname();
      obj.defaultdetectpat = [obj.defaulttrkpat '_tracklet'];
      obj.initData(toTrack);      
      obj.createGUI();
    end
    
    function toTrack = run(obj)
      uiwait(obj.gdata.fig);
      toTrack = obj.toTrack;
    end
    
    function initData(obj,toTrack)
      obj.toTrack = toTrack;
      if ~isfield(obj.toTrack,'movfiles'),
        obj.toTrack.movfiles = {};
      end
      if ~isfield(obj.toTrack,'trkfiles'),
        obj.toTrack.trkfiles = {};
      end
      if ~isfield(obj.toTrack,'trxfiles'),
        obj.toTrack.trxfiles = {};
      end
      if ~isfield(obj.toTrack,'cropRois'),
        obj.toTrack.cropRois = {};
      end
      if ~isfield(obj.toTrack,'calibrationfiles'),
        obj.toTrack.calibrationfiles = {};
      end
      if ~isfield(obj.toTrack,'targets'),
        obj.toTrack.targets = {};
      end
      if ~isfield(obj.toTrack,'f0s'),
        obj.toTrack.f0s = {};
      end
      if ~isfield(obj.toTrack,'f1s'),
        obj.toTrack.f1s = {};
      end
      obj.nmovies = size(obj.toTrack.movfiles,1);
    end
    
    function createGUI(obj)
      
      figname = 'Select movies to track';
      
      % compute the figure size
      units = get(obj.hParent,'units');
      set(obj.hParent,'Units','normalized');
      mainfigpos = get(obj.hParent,'Position');
      set(obj.hParent,'Units',units);
      figsz = [.4,.4]; % width, height
      % AL20201116: multimonitor setups, figsz can be bigger than
      % mainfogpos(3:4) which leads to 'huge' TrackBatchGUI pane. Cap size
      % of TrackBatchGUI.
      figsz = min(figsz,mainfigpos(3:4));
      mainfigctr = mainfigpos([1,2]) + mainfigpos([3,4])/2;
      figpos = [mainfigctr-figsz/2,figsz];
      % to do: make sure this is on the screen
      
      border = .025;
      %colw = .3;
      filebuttonw = .05;
      rowh = .05;
      rowborder = .01;
      colborder = .005;
      backgroundcolor = [0,0,0];
      editfilecolor = [.2,.2,.2];
      deletebuttoncolor = [.7,0,0];
      pagebuttoncolor = [0,0,0];
      addbuttoncolor = [.7,0,.7];
      
      colw = ((1-2*border) - (filebuttonw+colborder)*2);
      if obj.isma
        colw = colw/3;
      else
        colw = colw/2;
      end
      
      allmovieh = 1 - 3*border - rowh*4 - 4*rowborder - border;
      obj.nmovies_per_page = floor(allmovieh/(rowh+rowborder))-3;
      %disp(obj.nmovies_per_page);
      obj.setNPages();
      moveditx = border;
      coltitley = 1-border-rowh;
      if obj.isma
        detecteditx = moveditx + colw + colborder;
        trkeditx = detecteditx + colw + colborder;
      else
        trkeditx = moveditx + colw + colborder;
      end
      detailsbuttonx = trkeditx+colw+colborder;
      deletebuttonx = detailsbuttonx+filebuttonw+colborder;
      rowys = coltitley - (rowh+rowborder)*(1:obj.nmovies_per_page);
 
      macroedity = rowys(end) - 1.5*(rowh+rowborder);
      hasTrx = obj.lObj.hasTrx;
      if hasTrx
        macroedity(2) = macroedity - (rowh+rowborder);
      end
      
      pagetextw = .1;
      pagebuttonstrs = {'|<','<','>','>|'};
      npagebuttons = numel(pagebuttonstrs);
      allpagebuttonws = npagebuttons*filebuttonw + npagebuttons*colborder + pagetextw;
      pagebuttonx1 = .5-allpagebuttonws/2;
      pagebuttonxsless = pagebuttonx1 + (filebuttonw+colborder)*[0,1];
      pagetextx = pagebuttonxsless(end)+filebuttonw+colborder;
      pagebuttonxsmore = pagetextx + pagetextw + colborder+(filebuttonw+colborder)*[0,1];
      pagebuttonxs = [pagebuttonxsless,pagebuttonxsmore];
      pagebuttony = macroedity(end) - 1.5*(rowh+rowborder);
      
%      macroedity = pagebuttony + rowh + 2*rowborder;
      
      addbuttonw = .30;
      addbuttonx = .5 - addbuttonw/2;
      addbuttony = pagebuttony - rowh - border;
      
      % save, track, cancel
      controlbuttonw = .15;
      if obj.isma
        controlbuttonstrs = {'Save','Load','Detect','Link','Track','Cancel'};
        controlbuttontags = {'save','load','detect','link','track','cancel'};
        controlbuttoncolors = ...
          [0,0,.8
          0,.7,.7
          0,.7,0
          0,.7,0
          0,.7,0
          .4,.4,.4];
        
      else
        controlbuttonstrs = {'Save','Load','Track','Cancel'};
        controlbuttontags = {'save','load','track','cancel'};
        controlbuttoncolors = ...
          [0,0,.8
          0,.7,.7
          0,.7,0
          .4,.4,.4];        
      end
      ncontrolbuttons = numel(controlbuttonstrs);
      allcontrolbuttonw = ncontrolbuttons*controlbuttonw + (ncontrolbuttons-1)*colborder;
      controlbuttonx1 = .5-allcontrolbuttonw/2;
      controlbuttonxs = controlbuttonx1 + (controlbuttonw+colborder)*(0:ncontrolbuttons-1);
      controlbuttony = border;
      
      defaultmovfiles = cell(1,obj.nmovies_per_page);
      defaultdetectfiles = cell(1,obj.nmovies_per_page);
      defaulttrkfiles = cell(1,obj.nmovies_per_page);
      for i = 1:obj.nmovies_per_page
        defaultmovfiles{i} = sprintf('pathtomovie%disveryveryvery/long/movieloc%d.avi',i,i);
        defaultdetectfiles{i} = sprintf('pathtotrk%disveryveryvery/long/outputtrkloc%d_tracklet.trk',i,i);
        defaulttrkfiles{i} = sprintf('pathtotrk%disveryveryvery/long/outputtrkloc%d.trk',i,i);
      end
      obj.gdata = struct;
      
      obj.gdata.fig = figure(...
        'menubar','none',...
        'toolbar','none',...
        'name',figname,...
        'NumberTitle','off',...
        'IntegerHandle','off',...
        'Tag','figure_SelectTrackBatch',...
        'color',backgroundcolor,...
        'units','normalized',...
        'position',figpos);
      %  'windowstyle','modal',...
      
      FONTSIZE = 14;
      FONTSIZESML = 12;
      obj.gdata.txt_movietitle = uicontrol('Style','text','String','Movie',...
        'ForegroundColor','w','BackgroundColor','k','FontWeight','bold',...
        'FontSize',FONTSIZE,...
        'Units','normalized','Position',[moveditx,coltitley,colw,rowh],...
        'Tag','Movie title');
      obj.gdata.txt_trktitle = uicontrol('Style','text','String','Output trk',...
        'ForegroundColor','w','BackgroundColor','k','FontWeight','bold',...
        'FontSize',FONTSIZE,...
        'Units','normalized','Position',[trkeditx,coltitley,colw,rowh],...
        'Tag','Trk title');
      if obj.isma
        obj.gdata.txt_trktitle = uicontrol('Style','text','String','Output Detect trk',...
        'ForegroundColor','w','BackgroundColor','k','FontWeight','bold',...
        'FontSize',FONTSIZE,...
        'Units','normalized','Position',[detecteditx,coltitley,colw,rowh],...
        'Tag','Detect title');        
      end
      movmacrodescs = Labeler.movTrkFileMacroDescs();
      smacros = obj.lObj.baseTrkFileMacros();
      macrotooltip = sprintf('Trkfile locations will be auto-generated based on this field. Available macros:\n');
      for f=fieldnames(movmacrodescs)',f=f{1}; %#ok<FXSET>
        macrotooltip = [macrotooltip sprintf('$%s -> %s\n',f,movmacrodescs.(f))]; %#ok<AGROW>
      end
      for f=fieldnames(smacros)',f=f{1}; %#ok<FXSET>
        macrotooltip = [macrotooltip sprintf('$%s -> %s\n',f,smacros.(f))]; %#ok<AGROW>
      end
      
      obj.gdata.txt_macro = uicontrol('Style','text','String','Default trkfile location',...
          'ForegroundColor','w','BackgroundColor','k','FontWeight','bold',...
          'FontSize',FONTSIZESML,...
          'Units','normalized','Position',[moveditx,macroedity(1),colw,rowh],...
          'Tag','txt_macro',...
          'Tooltip', macrotooltip);
      obj.gdata.edit_macro = uicontrol('Style','edit',...
          'String',obj.defaulttrkpat,...
          'ForegroundColor','w','BackgroundColor',editfilecolor,'FontWeight','normal',...
          'Units','normalized','Enable','on','Position',[trkeditx,macroedity(1),colw,rowh],...
          'Tag','edit_macro',...
          'Callback',@(h,e) obj.macro_changed(h,e),...
          'Tooltip', macrotooltip);
      if hasTrx
        obj.gdata.txt_macro = uicontrol('Style','text','String','Default trx location',...
          'ForegroundColor','w','BackgroundColor','k','FontWeight','bold',...
          'FontSize',FONTSIZESML,...
          'Units','normalized','Position',[moveditx,macroedity(2),colw,rowh],...
          'Tag','txt_macro_trx',...
          'Tooltip', macrotooltip);
        obj.gdata.edit_macro = uicontrol('Style','edit',...
          'String',obj.defaulttrxpat,...
          'ForegroundColor','w','BackgroundColor',editfilecolor,'FontWeight','normal',...
          'Units','normalized','Enable','on','Position',[trkeditx,macroedity(2),colw,rowh],...
          'Tag','edit_macro_trx',...
          'Callback',@(h,e) obj.macro_changed_trx(h,e),...
          'Tooltip', macrotooltip);
      end
%        obj.gdata.apply_macro = uicontrol('Style','pushbutton','String','Apply',...
%           'ForegroundColor','w','BackgroundColor',pagebuttoncolor,'FontWeight','bold',...
%           'Units','normalized','Enable','on','Position',[detailsbuttonx,macroedity,2*filebuttonw+colborder,rowh],...
%           'Tag',sprintf('pushbutton_page%d',i),'UserData',i,...
%           'Callback',@(h,e) obj.apply_macro(h,e),...
%           'Tooltip', 'Apply changes');
      for i = 1:obj.nmovies_per_page,
        visible = 'off';
        movfilecurr = defaultmovfiles{i};
        trkfilecurr = defaulttrkfiles{i};
        detectfilecurr = defaultdetectfiles{i};
        obj.gdata.edit_movie(i) = uicontrol('Style','edit','String',movfilecurr,...
          'ForegroundColor','w','BackgroundColor',editfilecolor,'FontWeight','normal',...
          'Units','normalized','Enable','on','Position',[moveditx,rowys(i),colw,rowh],...
          'Tag',sprintf('edit_movie%d',i),'UserData',i,...
          'HorizontalAlignment','right','Visible',visible,...
          'Callback',@(h,e) obj.edit_movie_Callback(h,e,i));
        obj.gdata.edit_trk(i) = uicontrol('Style','edit','String',trkfilecurr,...
          'ForegroundColor','w','BackgroundColor',editfilecolor,'FontWeight','normal',...
          'Units','normalized','Enable','on','Position',[trkeditx,rowys(i),colw,rowh],...
          'Tag',sprintf('edit_trk%d',i),'UserData',i,...
          'HorizontalAlignment','right','Visible',visible,...
          'Callback',@(h,e) obj.edit_trk_Callback(h,e,i));
        if obj.isma
          obj.gdata.edit_detect(i) = uicontrol('Style','edit','String',detectfilecurr,...
          'ForegroundColor','w','BackgroundColor',editfilecolor,'FontWeight','normal',...
          'Units','normalized','Enable','on','Position',[detecteditx,rowys(i),colw,rowh],...
          'Tag',sprintf('edit_detect%d',i),'UserData',i,...
          'HorizontalAlignment','right','Visible',visible,...
          'Callback',@(h,e) obj.edit_detect_Callback(h,e,i));
        end
        obj.gdata.button_details(i) = uicontrol('Style','pushbutton','String','...',...
          'ForegroundColor','w','BackgroundColor',editfilecolor,'FontWeight','normal',...
          'Units','normalized','Enable','on','Position',[detailsbuttonx,rowys(i),filebuttonw,rowh],...
          'String','...','Tag',sprintf('pushbutton_details%d',i),'UserData',i,...
          'Visible',visible,...
          'Callback',@(h,e) obj.pb_details_Callback(h,e,i));
        obj.gdata.button_delete(i) = uicontrol('Style','pushbutton','String','-',...
          'ForegroundColor','w','BackgroundColor',deletebuttoncolor,'FontWeight','bold',...
          'Units','normalized','Enable','on','Position',[deletebuttonx,rowys(i),filebuttonw,rowh],...
          'String','-','Tag',sprintf('pushbutton_delete%d',i),'UserData',i,...
          'ToolTip','Remove this movie from list',...
          'Visible',visible,...
          'Callback',@(h,e) obj.pb_delete_Callback(h,e,i));
      end
      
      for i = 1:npagebuttons,
        obj.gdata.button_page(i) = uicontrol('Style','pushbutton','String',pagebuttonstrs{i},...
          'ForegroundColor','w','BackgroundColor',pagebuttoncolor,'FontWeight','bold',...
          'Units','normalized','Enable','on','Position',[pagebuttonxs(i),pagebuttony,filebuttonw,rowh],...
          'Tag',sprintf('pushbutton_page%d',i),'UserData',i,...
          'Callback',@(h,e) obj.pb_page_Callback(h,e,pagebuttonstrs{i}));
      end
      obj.gdata.text_page = uicontrol('Style','text','String',sprintf('Page %d/%d',obj.page,obj.npages),...
        'ForegroundColor','w','BackgroundColor',backgroundcolor,'FontWeight','normal',...
        'FontSize',FONTSIZESML,...
        'Units','normalized','Enable','on','Position',[pagetextx,pagebuttony,pagetextw,rowh],...
        'Tag','text_page','HorizontalAlignment','center');
      obj.gdata.button_add = uicontrol('Style','pushbutton','String','Add movie',...
        'ForegroundColor','w','BackgroundColor',addbuttoncolor,...
        'FontWeight','bold','FontSize',FONTSIZE,...
        'Units','normalized','Enable','on','Position',[addbuttonx,addbuttony,addbuttonw,rowh],...
        'Tag','pushbutton_add',...
        'Callback',@(h,e) obj.pb_add_Callback(h,e,[]));
      
      for i = 1:ncontrolbuttons,
        obj.gdata.button_control(i) = uicontrol('Style','pushbutton','String',controlbuttonstrs{i},...
          'ForegroundColor','w','BackgroundColor',controlbuttoncolors(i,:),...
          'FontWeight','bold','FontSize',FONTSIZE,...
          'Units','normalized','Enable','on','Position',[controlbuttonxs(i),controlbuttony,controlbuttonw,rowh],...
          'Tag',sprintf('controlbutton_%s',controlbuttontags{i}),'UserData',i,...
          'Callback',@(h,e) obj.pb_control_Callback(h,e,controlbuttontags{i}));
      end
      obj.updateMovieList();

    end
     
    
    function updateMovieList(obj)
      
      for i = 1:obj.nmovies_per_page,
        moviei = obj.item2MovieIdx(i);
        if moviei <= obj.nmovies,
          movfilecurr = obj.toTrack.movfiles{moviei,1};
          trkfilecurr = obj.toTrack.trkfiles{moviei,1};
          if obj.isma,
            detectfilecurr = obj.toTrack.detectfiles{moviei,1};
          end
          visible = 'on';
        else
          movfilecurr = '';
          trkfilecurr = '';
          detectfilecurr = '';
          visible = 'off';
        end
        if obj.page == 1 && i == 1,
          visible = 'on';
        end
        set(obj.gdata.edit_movie(i),'String',movfilecurr);
        set(obj.gdata.edit_trk(i),'String',trkfilecurr);
        if obj.isma
          set(obj.gdata.edit_detect(i),'String',detectfilecurr);
        end
        obj.setRowVisible(i,visible);
        set([obj.gdata.edit_movie(i),obj.gdata.edit_trk(i),...
          obj.gdata.button_details(i),obj.gdata.button_delete(i)],'Visible',visible);
      end
        
      set(obj.gdata.text_page,'String',sprintf('Page %d/%d',obj.page,obj.npages));
      
    end
    function setRowVisible(obj,i,visible)
      set([obj.gdata.edit_movie(i),obj.gdata.edit_trk(i),...
        obj.gdata.button_details(i),obj.gdata.button_delete(i)],'Visible',visible);
      if obj.isma
              set(obj.gdata.edit_detect(i),'Visible',visible);
      end
    end
    
    function moviei = item2MovieIdx(obj,itemi)
      moviei = (obj.page-1)*obj.nmovies_per_page + itemi;
    end
    function itemi = movie2ItemIdx(obj,moviei)
      itemi = moviei - (obj.page-1)*obj.nmovies_per_page;
      if itemi <= 0 || itemi > obj.nmovies_per_page,
        itemi = [];
      end
    end
    function movdata = getMovData(obj,moviei)
      movdata = struct;
      if moviei <= obj.nmovies,
        movdata.movfiles = obj.toTrack.movfiles(moviei,:);
        movdata.trkfiles = obj.toTrack.trkfiles(moviei,:);
        movdata.trxfiles = obj.toTrack.trxfiles(moviei,:);
        movdata.calibrationfiles = obj.toTrack.calibrationfiles{moviei};
        movdata.cropRois = obj.toTrack.cropRois(moviei,:);
        movdata.targets = obj.toTrack.targets{moviei};
        movdata.f0s = obj.toTrack.f0s(moviei);
        if obj.isma
          movdata.detectfiles = obj.toTrack.detectfiles(moviei,:);
        else
          movdata.detectfiles = [];
        end
        if iscell(movdata.f0s),
          movdata.f0s = movdata.f0s{1};
        end
        movdata.f1s = obj.toTrack.f1s(moviei);
        if iscell(movdata.f1s),
          movdata.f1s = movdata.f1s{1};
        end
      end
    end
    function setMovData(obj,moviei,movdata)
      nview = obj.lObj.nview;
      obj.toTrack.movfiles(moviei,:) = movdata.movfiles;
      obj.toTrack.trkfiles(moviei,:) = movdata.trkfiles;
      if isfield(movdata,'trxfiles'),
        obj.toTrack.trxfiles(moviei,:) = movdata.trxfiles;
      else
        obj.toTrack.trxfiles(moviei,:) = repmat({''},[1,nview]);
      end
      if obj.isma
        obj.toTrack.detectfiles(moviei,:) = movdata.detectfiles;
      else
        obj.toTrack.detectfiles(moviei,:) = repmat({''},[1,nview]);
      end
      if isfield(movdata,'calibrationfiles'),
        obj.toTrack.calibrationfiles{moviei,1} = movdata.calibrationfiles;
      else
        obj.toTrack.calibrationfiles{moviei,1} = '';
      end
      if isfield(movdata,'cropRois') && ~isempty(movdata.cropRois),
        obj.toTrack.cropRois(moviei,:) = movdata.cropRois;
      else
        obj.toTrack.cropRois(moviei,:) = repmat({[]},[1,nview]);
      end
      if isfield(movdata,'targets'),
        obj.toTrack.targets{moviei,1} = movdata.targets;
      else
        obj.toTrack.targets{moviei,1} = [];
      end
      if isfield(movdata,'f0s') && ~isempty(movdata.f0s),
        obj.toTrack.f0s{moviei,1} = movdata.f0s;
      else
        obj.toTrack.f0s(moviei,1) = {1};
      end
      if isfield(movdata,'f1s') && ~isempty(movdata.f1s),
        obj.toTrack.f1s{moviei,1} = movdata.f1s;
      else
        obj.toTrack.f1s(moviei,1) = {inf};
      end
% MK 20200711 - This doesn't make sense so commenting it out      
%       if moviei < obj.nmovies,
%         obj.nmovies = moviei;
%       end
      obj.needsSave = true;
      
      itemi = obj.movie2ItemIdx(moviei);
      if isempty(itemi),
        return;
      end
      set(obj.gdata.edit_movie(itemi),'String',obj.toTrack.movfiles{moviei,1});
      set(obj.gdata.edit_trk(itemi),'String',obj.toTrack.trkfiles{moviei,1});
      if obj.isma
        set(obj.gdata.edit_detect(itemi),'String',obj.toTrack.detectfiles{moviei,1});
      end
      obj.setRowVisible(itemi,'on');      
    end
    function pb_details_Callback(obj,h,e,itemi) %#ok<*INUSL>
      obj.setBusy();
      moviei = obj.item2MovieIdx(itemi);
      movdata = obj.getMovData(moviei);      
      if obj.isma
        movdetailsobj = SpecifyMovieToTrackGUI(obj.lObj,obj.gdata.fig,...
          movdata,'defaulttrkpat',obj.defaulttrkpat,...
          'defaultdetectpat',obj.defaultdetectpat,...
          'detailed_options',false);
      else
        movdetailsobj = SpecifyMovieToTrackGUI(obj.lObj,obj.gdata.fig,...
        movdata,'defaulttrkpat',obj.defaulttrkpat,...
        'defaulttrxpat',obj.defaulttrxpat);
      end
      [movdataout,dostore] = movdetailsobj.run();
      if dostore,
        obj.setMovData(moviei,movdataout);
        % should only happen when using the first item
        if moviei > obj.nmovies,
          obj.nmovies = moviei;
        end
      end
      obj.setNotBusy();
    end
    function pb_delete_Callback(obj,h,e,itemi)
      moviei = obj.item2MovieIdx(itemi);
      if itemi == 1 && moviei > obj.nmovies,
        return;
      end
      
      obj.toTrack.movfiles(moviei,:) = [];
      obj.toTrack.trkfiles(moviei,:) = [];
      obj.toTrack.trxfiles(moviei,:) = [];
      if obj.isma
        obj.toTrack.detectfiles(moviei,:) = [];
      end
      obj.toTrack.calibrationfiles(moviei) = [];
      obj.toTrack.targets(moviei) = [];
      obj.toTrack.f0s(moviei) = [];
      obj.toTrack.f1s(moviei) = [];
      obj.toTrack.cropRois(moviei,:) = [];
      obj.nmovies = obj.nmovies - 1;
      obj.setNPages();
      if obj.page > obj.npages,
        obj.page = obj.npages;
      end
      obj.updateMovieList();
      obj.needsSave = true;      
    end
          
    function pb_add_Callback(obj,h,e,movdat) %#ok<*INUSD>
      if obj.isma
        movdetailsobj = SpecifyMovieToTrackGUI(obj.lObj,obj.gdata.fig,...
        movdat,'defaulttrkpat',obj.defaulttrkpat,...
        'defaultdetectpat',obj.defaultdetectpat);
      else
      movdetailsobj = SpecifyMovieToTrackGUI(obj.lObj,obj.gdata.fig,...
        movdat,'defaulttrkpat',obj.defaulttrkpat,...
        'defaulttrxpat',obj.defaulttrxpat);
      end
      movdataout = movdetailsobj.run();
      if ~isfield(movdataout,'movfiles') || isempty(movdataout.movfiles) || ...
          isempty(movdataout.movfiles{1}),
        return;
      end
      obj.nmovies = obj.nmovies + 1;
      moviei = obj.nmovies;
      obj.setMovData(moviei,movdataout);
      obj.setNPages();
      if obj.page < obj.npages,
        obj.page = obj.npages;
        obj.updateMovieList();
      end
    end
    function setNPages(obj)
      obj.npages = ceil(max(1,obj.nmovies) / obj.nmovies_per_page);
    end
    function setPage(obj,page)
      obj.page = page;
      obj.updateMovieList();
    end
    
    function pb_page_Callback(obj,h,e,key)
      switch key,
        case '|<',
          obj.setPage(1);
        case '>|',
          obj.setPage(obj.npages);
        case '<',
          if obj.page > 1,
            obj.setPage(obj.page-1);
          end
        case '>',
          if obj.page < obj.npages,
            obj.setPage(obj.page+1);
          end
      end
    end
    
    function pb_control_Callback(obj,h,e,tag)
      switch tag,
        case 'cancel',
          delete(obj.gdata.fig);
        case {'load','save'},
          if strcmpi(tag,'save'),
            if obj.nmovies == 0,
              uiwait(errordlg('No movies selected.'));
              return;
            end
          end
          obj.loadOrSaveToFile(tag);
        case {'track','link','detect'}
          if obj.nmovies == 0,
            uiwait(errordlg('No movies selected.'));
            return;
          end
          if obj.needsSave
            res = questdlg('Save list of movies to track for reference?');
            if strcmpi(res,'Cancel'),
              return;
            elseif strcmpi(res,'Yes'),
              success = obj.loadOrSaveToFile('save');
              if ~success,
                return;
              end
            end
          end
          trackBatch('lObj',obj.lObj,'toTrack',obj.toTrack,'track_type',tag);
          delete(obj.gdata.fig);
        otherwise
          error('Callback for %s not implemented',tag);
      end
    end
    
    function success = loadOrSaveToFile(obj,tag)
      persistent lastpath;
      if isempty(lastpath),
        lastpath = '';
      end
      success = false;
      if strcmpi(tag,'load'),
        [filename,pathname] = uigetfile('*.json','Load list of movies to track from file',lastpath);
      else
        [filename,pathname] = uiputfile('*.json','Save list of movies to track from file',lastpath);
      end
      if ~ischar(filename),
        return;
      end
      jsonfile = fullfile(pathname,filename);
      lastpath = jsonfile;
      if strcmpi(tag,'load'),
        try
          toTrack = parseToTrackJSON(jsonfile,obj.lObj); %#ok<*PROPLC>
        catch ME,
          warning('Error loading from jsonfile %s:\n%s',jsonfile,getReport(ME));
          uiwait(errordlg(sprintf('Could not load from jsonfile:\n%s',...
            getReport(ME,'basic','hyperlinks','off'))));
          return;
        end
        % cropRois is read in as a nview x 4 matrix
        if isfield(toTrack,'cropRois'),
          if size(toTrack.cropRois,2) < obj.lObj.nview && ...
              size(toTrack.cropRois,2) == 1,
            cropRois = toTrack.cropRois;
            toTrack.cropRois = cell(size(cropRois,1),obj.lObj.nview);
            for i = 1:size(cropRois,1),
              if size(cropRois{i},1) == obj.lObj.nview,
                for j = 1:obj.lObj.nview,
                  toTrack.cropRois{i,j} = cropRois{i}(j,:);
                end
              end
            end
          end
        end
        obj.toTrack = toTrack;
        obj.nmovies = size(obj.toTrack.movfiles,1);
        obj.page = 1;
        obj.setNPages();
        obj.updateMovieList();
      else
        writeToTrackJSON(obj.toTrack,jsonfile);        
      end
      obj.needsSave = false;
      success = true;
    end

    function setBusy(obj,val)
      if nargin < 2,
        val = true;
      end
      obj.isbusy = val;
      controls = [obj.gdata.button_delete,obj.gdata.button_details,...
        obj.gdata.button_control,obj.gdata.button_page,obj.gdata.button_add];
      isvisible = strcmpi(get(controls,'visible'),'on');
      if val,
        set(controls(isvisible),'Enable','off');
      else
        set(controls(isvisible),'Enable','on');
      end
    end
    function setNotBusy(obj)
      obj.setBusy(false);
    end
    function edit_movie_Callback(obj,h,e,itemi)
      moviei = obj.item2MovieIdx(itemi);
      movie = h.String;

      defaulttrk = obj.defaulttrkpat;
      tfgentrk = ~isempty(defaulttrk);
      if tfgentrk
        trk = obj.genTrkfile(movie,defaulttrk);
      end
      defaulttrx = obj.defaulttrxpat;
      tfgentrx = obj.lObj.hasTrx && ~isempty(defaulttrx);
      if tfgentrx
        trx = obj.genTrkfile(movie,defaulttrx,'enforceExt',false);
      end
      defaultdetect = obj.defaultdetectpat;
      tfgendetect = obj.isma && ~isempty(defaultdetect);
      if tfgendetect
        detect = obj.genTrkfile(movie,defaultdetect);
      end
      
      if moviei>obj.nmovies
        nvw = obj.lObj.nview;
        movfiles = cell(1,nvw);
        movfiles{1} = movie;
        movdata = struct('movfiles',{movfiles});
        if tfgentrk
          trkfiles = cell(1,nvw);
          trkfiles{1} = trk;
          movdata.trkfiles = trkfiles;
        end
        if tfgentrx
          trxfiles = cell(1,nvw);
          trxfiles{1} = trx;
          movdata.trxfiles = trxfiles;
        end
        if tfgendetect
          detectfiles = cell(1,nvw);
          detectfiles{1} = detect;
          movdata.detectfiles = detectfiles;
        end
        obj.pb_add_Callback([],[],movdata);
      else
        obj.toTrack.movfiles{moviei,1} = movie;
        if tfgentrk
          obj.toTrack.trkfiles{moviei,1} = trk;
        end
        if tfgentrx
          obj.toTrack.trxfiles{moviei,1} = trx;
        end
        if tfgendetect
          obj.toTrack.detectfiles{moviei,1} = detect;
        end
        if tfgentrk || tfgentrx || tfgendetect
          obj.updateMovieList();
        end
        % mutating .toTrack outside setMovData
        obj.needsSave = true;
      end
    end
    function edit_trk_Callback(obj,h,e,itemi)
      moviei = obj.item2MovieIdx(itemi);
      if moviei>obj.nmovies
        obj.pb_add_Callback([],[],[]);
        return;
      end
      trk = h.String;
      obj.toTrack.trkfiles{moviei,1} = trk;
      % mutating .toTrack outside setMovData
      obj.needsSave = true;
    end
    function edit_detect_Callback(obj,h,e,itemi)
      moviei = obj.item2MovieIdx(itemi);
      if moviei>obj.nmovies
        obj.pb_add_Callback([],[],[]);
        return;
      end
      detect = h.String;
      obj.toTrack.detectfiles{moviei,1} = detect;
      % mutating .toTrack outside setMovData
      obj.needsSave = true;
    end
    function trk = genTrkfile(obj,movie,defaulttrk,varargin)  
      if isempty(movie)
        trk = '';
      else
        trk = Labeler.genTrkFileName(defaulttrk,...
          obj.lObj.baseTrkFileMacros(),movie,varargin{:});
      end
    end
    
    function macro_changed(obj,h,e)
      trkpatnew = strtrim(h.String);
      if strcmp(trkpatnew,obj.defaulttrkpat)
        return;
      end
      
      if ~isempty(trkpatnew) && obj.nmovies>0
        btn = questdlg('Apply to existing movies?','Default trkfile changed',...
          'Yes','No','Cancel','Yes');
        if isempty(btn)
          btn = 'Cancel';
        end
        switch btn
          case 'Yes'
            obj.defaulttrkpat = trkpatnew;
            obj.defaultdetectpat = [obj.defaulttrkpat '_tracklet'];
            obj.apply_macro_allmovies();
          case 'No'
            obj.defaulttrkpat = trkpatnew;
            obj.defaultdetectpat = [obj.defaulttrkpat '_tracklet'];
          case 'Cancel'
            % revert
            h.String = obj.defaulttrkpat;            
        end
      else
        obj.defaulttrkpat = trkpatnew;
        obj.defaultdetectpat = [obj.defaulttrkpat '_tracklet'];

      end
    end
    function apply_macro_allmovies(obj)
      defaulttrk = obj.defaulttrkpat;
      for moviei = 1:obj.nmovies
        for view = 1:obj.lObj.nview
          cur_m = obj.toTrack.movfiles{moviei,view};
          trk = obj.genTrkfile(cur_m,defaulttrk);
          obj.toTrack.trkfiles{moviei,view} = trk;
          if obj.lObj.maIsMA
            trk = obj.genTrkfile(cur_m,obj.defaultdetectpat);
            obj.toTrack.detectfiles{moviei,view} = trk;
          end
        end
      end
      obj.updateMovieList();
      % mutating .toTrack outside setMovData
      obj.needsSave = true;
    end
    function macro_changed_trx(obj,h,e)
      trxpat = strtrim(h.String);
      if strcmp(trxpat,obj.defaulttrxpat)
        return;
      end
      obj.defaulttrxpat = trxpat;
    end

  end
end

