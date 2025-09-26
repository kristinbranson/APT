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
    
    % Store original (untruncated) data for dynamic resizing
    originalMovieFiles = {};
    originalTrkFiles = {};
    originalTrxFiles = {};
    originalDetectFiles = {};
    hasTrx = false;
    
    % Path display mode: true = show path ends, false = show path starts
    showPathEnds = true;
    
    % Note: Linking options are stored in obj.toTrack.link_type and obj.toTrack.id_maintain_identity
  end
  
  methods
    function obj = TrackBatchGUI(lObj,varargin)
      obj.lObj = lObj;
      obj.isma = lObj.maIsMA;
      obj.hParent = obj.lObj.gdata.mainFigure_;      
      [toTrack,debug] = myparse(varargin,'toTrack',struct,'debug',false);
      
      obj.defaulttrkpat = lObj.defaultExportTrkRawname();
      obj.defaultdetectpat = [obj.defaulttrkpat '.tracklet'];
      obj.initData(toTrack);      
      obj.createGUI();

      % This is to resize once the figure is rendered.
      getframe(obj.gdata.fig);
      obj.figureResizeCallback(obj.gdata.fig,[]);
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
      if ~isfield(obj.toTrack,'link_type'),
        obj.toTrack.link_type = 'simple';
      end
      if ~isfield(obj.toTrack,'id_maintain_identity'),
        obj.toTrack.id_maintain_identity = false;
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
      colw = colw/2;
      
      allmovieh = 1 - 3*border - rowh*4 - 4*rowborder - border;
      obj.nmovies_per_page = floor(allmovieh/(rowh+rowborder))-3;
      %disp(obj.nmovies_per_page);
      obj.setNPages();
      moveditx = border;
      coltitley = 1-border-rowh;
      % if obj.isma
      %   detecteditx = moveditx + colw + colborder;
      %   trkeditx = detecteditx + colw + colborder;
      % else
      trkeditx = moveditx + colw + colborder;
      % end
      detailsbuttonx = trkeditx+colw+colborder;
      deletebuttonx = detailsbuttonx+filebuttonw+colborder;
      rowys = coltitley - (rowh+rowborder)*(1:obj.nmovies_per_page);
 
      macroedity = rowys(end) - 1.5*(rowh+rowborder);
      hasTrx = obj.lObj.hasTrx;
      obj.hasTrx = hasTrx;
      if hasTrx
        macroedity(2) = macroedity - (rowh+rowborder);
        ncol = 3;
      else
        ncol = 2;
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
      controlbuttonstrs = {'Track','Cancel'};
      controlbuttontags = {'track','cancel'};
      controlbuttoncolors = ...
        [0,.7,0
        .4,.4,.4];        
      ncontrolbuttons = numel(controlbuttonstrs);
      allcontrolbuttonw = ncontrolbuttons*controlbuttonw + (ncontrolbuttons-1)*colborder;
      controlbuttonx1 = .5-allcontrolbuttonw/2;
      controlbuttonxs = controlbuttonx1 + (controlbuttonw+colborder)*(0:ncontrolbuttons-1);
      controlbuttony = border;
      
      defaultmovfiles = cell(1,obj.nmovies_per_page);
      defaulttrkfiles = cell(1,obj.nmovies_per_page);
      defaulttrxfiles = cell(1,obj.nmovies_per_page);
      for i = 1:obj.nmovies_per_page
        defaultmovfiles{i} = sprintf('pathtomovie%disveryveryvery/long/movieloc%d.avi',i,i);
        defaulttrkfiles{i} = sprintf('pathtotrk%disveryveryvery/long/outputtrkloc%d.trk',i,i);
        if hasTrx
          defaulttrxfiles{i} = sprintf('pathtotrx%disveryveryvery/long/trxloc%d.mat',i,i);
        end
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
        'position',figpos,...
        'ResizeFcn',@(src,evt) obj.figureResizeCallback(src,evt),...
        'WindowButtonDownFcn',@(src,evt) obj.windowButtonDownFcn(src,evt));
      %  'windowstyle','modal',...

      rows =  {'1x',40,40,40};
      if hasTrx,
        rows{end+1} = 40;
      end
      % Add row for identity linking checkboxes (multi-animal projects only)
      if obj.lObj.maIsMA
        rows{end+1} = 40;
      end
      grid = uigridlayout(obj.gdata.fig,'RowHeight',rows,...
        'ColumnWidth', {'1x'},'BackgroundColor',backgroundcolor,'RowSpacing',10);
      obj.gdata.grid = grid;
      
      mov_row = repmat({'1x'},1,obj.nmovies_per_page+1);
      mov_col = [repmat({'1x'},1,ncol), {40, 40}]';
      edit_grid = uigridlayout(grid,'RowHeight',mov_row,'ColumnWidth',mov_col,...
        'BackgroundColor',backgroundcolor,'Padding',[0 0 0 0],'RowSpacing',3);
      edit_grid.Layout.Row = 1;
      edit_grid.Layout.Column = [1 2];
      obj.gdata.edit_grid = edit_grid;

      FONTSIZE = 14;
      FONTSIZESML = 12;
      obj.gdata.txt_movietitle = uilabel(edit_grid,'Text','Movie',...
        'FontColor','w','BackgroundColor','k','FontWeight','bold',...
        'FontSize',FONTSIZE,'HorizontalAlignment','center',...
        'Tag','Movie title');
      obj.gdata.txt_movietitle.Layout.Row=1;
      obj.gdata.txt_movietitle.Layout.Column=1;
      if hasTrx
        obj.gdata.txt_trxtitle = uilabel(edit_grid,'Text','Trx File',...
          'FontColor','w','BackgroundColor','k','FontWeight','bold',...
          'FontSize',FONTSIZE,'HorizontalAlignment','center',...
          'Tag','Trx title');
        obj.gdata.txt_trxtitle.Layout.Row=1;
        obj.gdata.txt_trxtitle.Layout.Column=2;
        nextcol = 3;
      else
        nextcol = 2;
      end

      obj.gdata.txt_trktitle = uilabel(edit_grid,'Text','Output trk',...
        'FontColor','w','BackgroundColor','k','FontWeight','bold',...
        'FontSize',FONTSIZE,'HorizontalAlignment','center',...
        'Tag','Trk title');
      obj.gdata.txt_trktitle.Layout.Row=1;
      obj.gdata.txt_trktitle.Layout.Column=nextcol;
      % if obj.isma
      %   obj.gdata.txt_detecttitle = uilabel(edit_grid,'Text','Output Detect trk',...
      %   'FontColor','w','BackgroundColor','k','FontWeight','bold',...
      %   'FontSize',FONTSIZE,...
      %   'Tag','Detect title');
      %   obj.gdata.txt_detecttitle.Layout.Row=1;
      %   obj.gdata.txt_detecttitle.Layout.Column=3;
      % end

      % Create button group for path display toggle buttons
      obj.gdata.bg_path = uibuttongroup(edit_grid,...
        'BackgroundColor','k',...
        'BorderType','none',...
        'SelectionChangedFcn',@(src,evt) obj.pathToggleChanged(src,evt));
      obj.gdata.bg_path.Layout.Row = 1;
      obj.gdata.bg_path.Layout.Column = nextcol + [1 2];

      % Load icon images
      leftAlignIcon = imread(fullfile(fileparts(mfilename('fullpath')), 'util', 'align_left.png'));
      rightAlignIcon = imread(fullfile(fileparts(mfilename('fullpath')), 'util', 'align_right.png'));
      if ndims(leftAlignIcon)==2
        leftAlignIcon = repmat(leftAlignIcon,[1 1 3]);
      end
      if ndims(rightAlignIcon)==2
        rightAlignIcon = repmat(rightAlignIcon,[1 1 3]);
      end

      % "Starts" toggle button (left side, above details column)
      obj.gdata.tb_path_starts = uitogglebutton(obj.gdata.bg_path,...
        'Text','',...
        'Icon',leftAlignIcon,...
        'Tooltip','Show path starts',...
        'FontColor','w','BackgroundColor',[1,1,1],...
        'FontWeight','bold','FontSize',FONTSIZESML,...
        'Value',~obj.showPathEnds,...
        'Tag','togglebutton_path_starts');

      % "Ends" toggle button (right side, above delete column)
      obj.gdata.tb_path_ends = uitogglebutton(obj.gdata.bg_path,...
        'Text','',...
        'Icon',rightAlignIcon,...
        'Tooltip','Show path ends',...
        'FontColor','w','BackgroundColor',[1,1,1],...
        'FontWeight','bold','FontSize',FONTSIZESML,...
        'Value',obj.showPathEnds,...
        'Tag','togglebutton_path_ends');

      % Position the toggle buttons initially
      obj.updatePathTogglePositions();

      movmacrodescs = Labeler.movTrkFileMacroDescs();
      smacros = obj.lObj.baseTrkFileMacros();
      macrotooltip = sprintf('Trkfile locations will be auto-generated based on this field. Available macros:\n');
      for f=fieldnames(movmacrodescs)',f=f{1}; %#ok<FXSET>
        macrotooltip = [macrotooltip sprintf('$%s -> %s\n',f,movmacrodescs.(f))]; %#ok<AGROW>
      end
      for f=fieldnames(smacros)',f=f{1}; %#ok<FXSET>
        macrotooltip = [macrotooltip sprintf('$%s -> %s\n',f,smacros.(f))]; %#ok<AGROW>
      end
      
      obj.gdata.txt_macro = uilabel(grid,'Text','Default trkfile location',...
          'FontColor','w','BackgroundColor','k','FontWeight','bold',...
          'FontSize',FONTSIZESML,'HorizontalAlignment','right',...          
          'Tag','txt_macro',...
          'Tooltip', macrotooltip);
      obj.gdata.txt_macro.Layout.Row = 2;
      obj.gdata.txt_macro.Layout.Column = 1;
      obj.gdata.edit_macro = uieditfield(grid,'Value',obj.defaulttrkpat,...
          'FontColor','w','BackgroundColor',editfilecolor,'FontWeight','normal',...
          'Enable','on','HorizontalAlignment','left',...
          'Tag','edit_macro',...
          'ValueChangedFcn',@(h,e) obj.macro_changed(h,e),...
          'Tooltip', macrotooltip);
      obj.gdata.edit_macro.Layout.Row = 2;
      obj.gdata.edit_macro.Layout.Column = 2;
      if obj.hasTrx
        obj.gdata.txt_trx = uilabel(grid,'Text','Default trx location',...
          'FontColor','w','BackgroundColor','k','FontWeight','bold',...
          'FontSize',FONTSIZESML,'HorizontalAlignment','right',...
          'Tag','txt_macro_trx',...
          'Tooltip', macrotooltip);
        obj.gdata.txt_trx.Layout.Row = 3;
        obj.gdata.txt_trx.Layout.Column = 1;

        obj.gdata.edit_trx = uieditfield(grid,...
          'Value',obj.defaulttrxpat,...
          'FontColor','w','BackgroundColor',editfilecolor,'FontWeight','normal',...
          'Enable','on','HorizontalAlignment','left',...
          'Tag','edit_macro_trx',...
          'ValueChangedFcn',@(h,e) obj.macro_changed_trx(h,e),...
          'Tooltip', macrotooltip);
        obj.gdata.edit_trx.Layout.Row = 3;
        obj.gdata.edit_trx.Layout.Column = 2;
      end
      
      % Add identity linking checkboxes for multi-animal projects
      if obj.lObj.maIsMA
        if obj.hasTrx
          nextRow = 4;
        else
          nextRow = 3;
        end
        
        % Create a sub-grid for the radio buttons and checkbox
        linking_grid = uigridlayout(grid,'RowHeight',{'1x'},'ColumnWidth',{'3x','1x'},...
          'BackgroundColor',backgroundcolor,'Padding',[0 0 0 0],'RowSpacing',5);
        linking_grid.Layout.Row = nextRow;
        linking_grid.Layout.Column = [1 2];
        obj.gdata.linking_grid = linking_grid;
        
        % Create button group for radio buttons
        obj.gdata.bg_linking = uibuttongroup(linking_grid,...
          'BackgroundColor',backgroundcolor,...
          'BorderType','none',...
          'SelectionChangedFcn',@(src,evt) obj.linkingTypeChanged(src,evt));
        obj.gdata.bg_linking.Layout.Row = 1;
        obj.gdata.bg_linking.Layout.Column = 1;
        
        % Radio buttons - positions will be calculated dynamically
        obj.gdata.rb_simple = uiradiobutton(obj.gdata.bg_linking,...
          'Text','Simple linking',...
          'FontSize',FONTSIZE,'FontColor','w',...
          'Position',[5 5 200 20],...
          'Value',strcmp(obj.toTrack.link_type,'simple'));

        obj.gdata.rb_motion = uiradiobutton(obj.gdata.bg_linking,...
          'Text','Motion Linking',...
          'FontSize',FONTSIZE,'FontColor','w',...
          'Position',[230 5 200 20],...
          'Value',strcmp(obj.toTrack.link_type,'motion'));

        obj.gdata.rb_identity = uiradiobutton(obj.gdata.bg_linking,...
          'Text','Identity linking',...
          'FontSize',FONTSIZE,'FontColor','w',...
          'Position',[455 5 200 20],...
          'Value',strcmp(obj.toTrack.link_type,'identity'));

        % Maintain identities checkbox
        obj.gdata.chk_maintain_identities = uicheckbox(linking_grid,...
          'Text','Maintain identities across videos',...
          'FontSize',FONTSIZE,'FontColor','w',...
          'Value',obj.toTrack.id_maintain_identity,...
          'Enable',strcmp(obj.toTrack.link_type,'identity'),...
          'ValueChangedFcn',@(h,e) obj.idMaintainIdentityChanged(h,e));
        obj.gdata.chk_maintain_identities.Layout.Row = 1;
        obj.gdata.chk_maintain_identities.Layout.Column = 2;
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
        obj.gdata.edit_movie(i) = uieditfield(edit_grid,...
          'Value',movfilecurr,...
          'FontColor','w','BackgroundColor',editfilecolor,'FontWeight','normal',...
          'Enable','on',...
          'Tag',sprintf('edit_movie%d',i),...
          'HorizontalAlignment','right','Visible',visible,...
          'ValueChangedFcn',@(h,e) obj.edit_movie_Callback(h,e,i),...
          'ValueChangingFcn',@(h,e) obj.expandEditFieldOnChange(h,e,i,'movie'));
        obj.gdata.edit_movie(i).Layout.Row = i+1;
        obj.gdata.edit_movie(i).Layout.Column = 1;

        if hasTrx
          trxfilecurr = defaulttrxfiles{i};
          obj.gdata.edit_trx(i) = uieditfield(edit_grid,...
            'Value',trxfilecurr,...
            'FontColor','w','BackgroundColor',editfilecolor,'FontWeight','normal',...
            'Enable','on',...
            'Tag',sprintf('edit_trx%d',i),...
            'HorizontalAlignment','right','Visible',visible,...
            'ValueChangedFcn',@(h,e) obj.edit_trx_Callback(h,e,i),...
            'ValueChangingFcn',@(h,e) obj.expandEditFieldOnChange(h,e,i,'trx'));
          obj.gdata.edit_trx(i).Layout.Row = i+1;
          obj.gdata.edit_trx(i).Layout.Column = 2;
        end

        obj.gdata.edit_trk(i) = uieditfield(edit_grid,...
          'Value',trkfilecurr,...
          'FontColor','w','BackgroundColor',editfilecolor,'FontWeight','normal',...
          'Enable','on',...
          'Tag',sprintf('edit_trk%d',i),...
          'HorizontalAlignment','right','Visible',visible,...
          'ValueChangedFcn',@(h,e) obj.edit_trk_Callback(h,e,i),...
          'ValueChangingFcn',@(h,e) obj.expandEditFieldOnChange(h,e,i,'trk'));
        obj.gdata.edit_trk(i).Layout.Row = i+1;
        obj.gdata.edit_trk(i).Layout.Column = 2+hasTrx;
        % if obj.isma
        %   obj.gdata.edit_detect(i) = uieditfield(obj.gdata.fig,'text',...
        %   'Value',detectfilecurr,...
        %   'FontColor','w','BackgroundColor',editfilecolor,'FontWeight','normal',...
        %   'Enable','on','Position',[detecteditx,rowys(i),colw,rowh],...
        %   'Tag',sprintf('edit_detect%d',i),...
        %   'HorizontalAlignment','right','Visible',visible,...
        %   'ValueChangedFcn',@(h,e) obj.edit_detect_Callback(h,e,i),...
        %   'ValueChangingFcn',@(h,e) obj.expandEditFieldOnChange(h,e,i,'detect'));
        % end
        obj.gdata.button_details(i) = uibutton(edit_grid,'Text','...',...
          'FontColor','w','BackgroundColor',editfilecolor,'FontWeight','normal',...
          'Enable','on',...
          'Tag',sprintf('pushbutton_details%d',i),...
          'Visible',visible,...
          'ButtonPushedFcn',@(h,e) obj.pb_details_Callback(h,e,i));
        obj.gdata.button_details(i).Layout.Row = i+1;
        obj.gdata.button_details(i).Layout.Column = 3+hasTrx;
        obj.gdata.button_delete(i) = uibutton(edit_grid,'Text','-',...
          'FontColor','w','BackgroundColor',deletebuttoncolor,'FontWeight','bold',...
          'Enable','on',...
          'Tag',sprintf('pushbutton_delete%d',i),'UserData',i,...
          'ToolTip','Remove this movie from list',...
          'Visible',visible,...
          'ButtonPushedFcn',@(h,e) obj.pb_delete_Callback(h,e,i));
        obj.gdata.button_delete(i).Layout.Row = i+1;
        obj.gdata.button_delete(i).Layout.Column = 4+hasTrx;
      end
      
      page_cols = repmat({50},[1 npagebuttons]);
      btn_cols = repmat({'1x'},[1 4]);
      page_cols = [btn_cols{:} page_cols{1:end/2} {80} page_cols{end/2+1:end}];
      n_page_cols = numel(page_cols);
      button_grid = uigridlayout(grid,'ColumnWidth',page_cols,'RowHeight',{'1x'},...
        'BackgroundColor',backgroundcolor,'Padding',[0 0 0 0],'RowSpacing',0);
      button_grid.Layout.Row = numel(grid.RowHeight)-1;
      button_grid.Layout.Column = [1 2];
      obj.gdata.text_page = uilabel(button_grid,'Text',sprintf('Page %d/%d',obj.page,obj.npages),...
        'FontColor','w','BackgroundColor',backgroundcolor,'FontWeight','normal',...
        'FontSize',FONTSIZESML,...
        'Enable','on',...
        'Tag','text_page','HorizontalAlignment','center');
      obj.gdata.text_page.Layout.Column = n_page_cols-5+npagebuttons/2+1;

      obj.gdata.button_save = uibutton(button_grid,'Text','Save',...
        'FontColor','w','BackgroundColor',[0,0,.8],...
        'FontWeight','bold','FontSize',FONTSIZE,...
        'Enable','on',...
        'Tag',sprintf('controlbutton_save'),'UserData',2,...
        'ButtonPushedFcn',@(h,e) obj.pb_control_Callback(h,e,'save'));
      obj.gdata.button_save.Layout.Column = 1;

      obj.gdata.button_load = uibutton(button_grid,'Text','Load',...
        'FontColor','w','BackgroundColor',[0,.7,.7],...
        'FontWeight','bold','FontSize',FONTSIZE,...
        'Enable','on',...
        'Tag',sprintf('controlbutton_load'),'UserData',3,...
        'ButtonPushedFcn',@(h,e) obj.pb_control_Callback(h,e,'load'));
      obj.gdata.button_load.Layout.Column = 2;

      obj.gdata.button_add = uibutton(button_grid,'Text','Add movie',...
        'FontColor','w','BackgroundColor',addbuttoncolor,...
        'FontWeight','bold','FontSize',FONTSIZE,...
        'Enable','on',...
        'Tag','pushbutton_add',...
        'ButtonPushedFcn',@(h,e) obj.pb_add_Callback(h,e,[]));
      obj.gdata.button_add.Layout.Column = 3;

      obj.gdata.button_add_list = uibutton(button_grid,'Text','Add List',...
        'FontColor','w','BackgroundColor',addbuttoncolor,...
        'FontWeight','bold','FontSize',FONTSIZE,...
        'Enable','on',...
        'Tag','pushbutton_add_list',...
        'ButtonPushedFcn',@(h,e) obj.pb_add_list_Callback(h,e));
      obj.gdata.button_add_list.Layout.Column = 4;

      for i = 1:npagebuttons,
        obj.gdata.button_page(i) = uibutton(button_grid,'Text',pagebuttonstrs{i},...
          'FontColor','w','BackgroundColor',pagebuttoncolor,'FontWeight','bold',...
          'Enable','on',...
          'Tag',sprintf('pushbutton_page%d',i),...
          'ButtonPushedFcn',@(h,e) obj.pb_page_Callback(h,e,pagebuttonstrs{i}));
        obj.gdata.button_page(i).Layout.Column = n_page_cols-5+i+floor(i/npagebuttons*2-0.5);
      end


      button_grid = uigridlayout(grid,'ColumnWidth',repmat({'1x'},[1 ncontrolbuttons]),...
        'BackgroundColor',backgroundcolor,'Padding',[0 0 0 0],'RowSpacing',0,'RowHeight',{'1x'});
      button_grid.Layout.Row = numel(grid.RowHeight);
      button_grid.Layout.Column = [1 2];
      for i = 1:ncontrolbuttons,
        obj.gdata.button_control(i) = uibutton(button_grid,'Text',controlbuttonstrs{i},...
          'FontColor','w','BackgroundColor',controlbuttoncolors(i,:),...
          'FontWeight','bold','FontSize',FONTSIZE,...
          'Enable','on',...
          'Tag',sprintf('controlbutton_%s',controlbuttontags{i}),'UserData',i,...
          'ButtonPushedFcn',@(h,e) obj.pb_control_Callback(h,e,controlbuttontags{i}));
        obj.gdata.button_control(i).Layout.Column = i;
      end
      obj.updateMovieList();
      
      % Update radio button positions after GUI is created
      % if obj.lObj.maIsMA
      %   obj.updateLinkingButtonPositions();
      % end

    end
     
    
    function updateMovieList(obj)
      
      % Store original data for dynamic truncation on resize
      obj.originalMovieFiles = cell(obj.nmovies, 1);
      obj.originalTrkFiles = cell(obj.nmovies, 1);
      if obj.hasTrx
        obj.originalTrxFiles = cell(obj.nmovies,1);
      end
      % if obj.isma
      %   obj.originalDetectFiles = cell(obj.nmovies, 1);
      % end
      
      for imov = 1:obj.nmovies
        obj.originalMovieFiles{imov} = obj.toTrack.movfiles{imov,1};
        obj.originalTrkFiles{imov} = obj.toTrack.trkfiles{imov,1};
        if obj.hasTrx
          obj.originalTrxFiles{imov} = obj.toTrack.trxfiles{imov,1};
        end
      end

      for i = 1:obj.nmovies_per_page,
        moviei = obj.item2MovieIdx(i);
        if moviei <= obj.nmovies,
          movfilecurr = obj.toTrack.movfiles{moviei,1};
          trkfilecurr = obj.toTrack.trkfiles{moviei,1};
          if obj.hasTrx
            trxfilecurr = obj.toTrack.trxfiles{moviei,1};
          end
          visible = 'on';
        else
          movfilecurr = '';
          trkfilecurr = '';
          if obj.hasTrx
            trxfilecurr = '';
          end
          visible = 'off';
        end
        if obj.page == 1 && i == 1,
          visible = 'on';
        end
        set(obj.gdata.edit_movie(i),'Value',movfilecurr);
        set(obj.gdata.edit_trk(i),'Value',trkfilecurr);
        if obj.hasTrx
          set(obj.gdata.edit_trx(i),'Value',trxfilecurr);
        end
        obj.setRowVisible(i,visible);
        set([obj.gdata.edit_movie(i),obj.gdata.edit_trk(i),...
          obj.gdata.button_details(i),obj.gdata.button_delete(i)],'Visible',visible);
        if obj.hasTrx
          set(obj.gdata.edit_trx(i),'Value',trxfilecurr,'Visible',visible);
        end
        
      end
        
      set(obj.gdata.text_page,'Text',sprintf('Page %d/%d',obj.page,obj.npages));
      
      % Apply current truncation mode to all visible paths
      obj.updateTruncatedFilePathsWithMode();
      
    end
    function setRowVisible(obj,i,visible)
      set([obj.gdata.edit_movie(i),obj.gdata.edit_trk(i),...
        obj.gdata.button_details(i),obj.gdata.button_delete(i)],'Visible',visible);
      if obj.hasTrx
        set(obj.gdata.edit_trx(i),'Visible',visible);
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
        % if obj.isma
        %   movdata.detectfiles = obj.toTrack.detectfiles(moviei,:);
        % else
        %   movdata.detectfiles = [];
        % end
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
      
      % Update original data storage for this movie
      if length(obj.originalMovieFiles) < moviei
        obj.originalMovieFiles{moviei,:} = [];
        obj.originalTrkFiles{moviei,:} = [];
        if obj.hasTrx
          obj.originalTrxFiles{moviei,:} = [];
        end
      end
      obj.originalMovieFiles{moviei,:} = obj.toTrack.movfiles{moviei,:};
      obj.originalTrkFiles{moviei,:} = obj.toTrack.trkfiles{moviei,:};
      if obj.hasTrx
        obj.originalTrxFiles{moviei,:} = obj.toTrack.trxfiles{moviei,1};
      end
      
      itemi = obj.movie2ItemIdx(moviei);
      if isempty(itemi),
        return;
      end
      set(obj.gdata.edit_movie(itemi),'Value',obj.toTrack.movfiles{moviei,1});
      set(obj.gdata.edit_trk(itemi),'Value',obj.toTrack.trkfiles{moviei,1});
      if obj.hasTrx
        set(obj.gdata.edit_trx(itemi),'Value',obj.toTrack.trxfiles{moviei,1});
      end
      obj.setRowVisible(itemi,'on');
      
      % Apply current truncation mode to this row
      obj.updateTruncatedFilePathsWithMode();      
    end
    function pb_details_Callback(obj,h,e,itemi) %#ok<*INUSL>
      obj.setBusy();
      moviei = obj.item2MovieIdx(itemi);
      movdata = obj.getMovData(moviei);      
      if obj.isma
        movdetailsobj = SpecifyMovieToTrackGUI(obj.lObj,obj.gdata.fig,...
          movdata,'defaulttrkpat',obj.defaulttrkpat,...
          'detailed_options',false);
          %'defaultdetectpat',obj.defaultdetectpat,...
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
      % if obj.isma
      %   obj.toTrack.detectfiles(moviei,:) = [];
      % end
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
        movdat,'defaulttrkpat',obj.defaulttrkpat,'detailed_options',false);
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

    function pb_add_list_Callback(obj,h,e) %#ok<*INUSD>
      % Callback for "Add List" button - loads a list of movies from a text file

      persistent lastpath;
      if isempty(lastpath)
        lastpath = '';
      end

      % Get file from user
      [filename,pathname] = uigetfile({'*.txt;*.list','Text files (*.txt, *.list)';...
                                       '*.*','All Files (*.*)'},...
                                       'Select movie list file',lastpath);
      if ~ischar(filename)
        return;
      end

      listfile = fullfile(pathname,filename);
      lastpath = pathname;

      % Read and parse the file
      try
        fid = fopen(listfile,'r');
        if fid == -1
          errordlg(sprintf('Could not open file: %s',listfile));
          return;
        end

        lines = {};
        while ~feof(fid)
          line = fgetl(fid);
          if ischar(line) && ~isempty(strtrim(line))
            lines{end+1} = strtrim(line); %#ok<AGROW>
          end
        end
        fclose(fid);

        if isempty(lines)
          msgbox('No valid entries found in the file.');
          return;
        end

        % Parse each line and add movies
        addedCount = 0;
        errorLines = {};

        for i = 1:length(lines)
          try
            movdataout = obj.parseMovieListLine(lines{i});
            if ~isempty(movdataout)
              obj.nmovies = obj.nmovies + 1;
              moviei = obj.nmovies;
              obj.setMovData(moviei,movdataout);
              addedCount = addedCount + 1;
            end
          catch ME
            errorLines{end+1} = sprintf('Line %d: %s (%s)',i,lines{i},ME.message); %#ok<AGROW>
          end
        end

        % Update display
        obj.setNPages();
        if obj.page < obj.npages
          obj.page = obj.npages;
          obj.updateMovieList();
        end

        % Show summary
        if addedCount > 0
          % msgStr = sprintf('Successfully added %d movies.',addedCount);
          % if ~isempty(errorLines)
          %   msgStr = [msgStr sprintf('\n\nErrors encountered:\n%s',strjoin(errorLines,'\n'))];
          % end
          % msgbox(msgStr,'Add List Results');
        else
          errordlg('No movies were added. Please check the file format.');
        end

      catch ME
        errordlg(sprintf('Error reading file: %s',ME.message));
      end
    end

    function movdataout = parseMovieListLine(obj,line)
      % Parse a single line from the movie list file
      % For single view: movie_file[,trx_file][,output_trk_file]
      % For multi-view: movie1,movie2,...[,trx1,trx2,...][,output1,output2,...]

      movdataout = [];

      % Split line by commas
      parts = strsplit(line,',');
      parts = cellfun(@strtrim,parts,'UniformOutput',false);

      % Remove empty parts
      parts = parts(~cellfun(@isempty,parts));

      if isempty(parts)
        return;
      end

      % Initialize movie data structure
      movdataout = struct();
      nviews = obj.lObj.nview;

      % Parse movie files (first nviews entries)
      if length(parts) < nviews
        error('Not enough movie files specified. Expected %d views, got %d files',nviews,length(parts));
      end

      movieFiles = parts(1:nviews);
      for i = 1:nviews
        if ~exist(movieFiles{i},'file')
          error('Movie file does not exist: %s',movieFiles{i});
        end
      end
      movdataout.movfiles = movieFiles;

      % Generate default trk files for each view
      defaultTrkFiles = cell(1,nviews);
      for i = 1:nviews
        defaultTrkFiles{i} = obj.genTrkfile(movieFiles{i},obj.defaulttrkpat);
      end
      movdataout.trkfiles = defaultTrkFiles;

      partIndex = nviews + 1;

      % Handle trx files if project requires them
      if obj.lObj.hasTrx
        if length(parts) >= partIndex + nviews - 1
          % Trx files specified for each view
          trxFiles = parts(partIndex:partIndex + nviews - 1);
          for i = 1:nviews
            if ~isempty(trxFiles{i}) && ~exist(trxFiles{i},'file')
              error('Trx file does not exist: %s',trxFiles{i});
            end
          end
          movdataout.trxfiles = trxFiles;
          partIndex = partIndex + nviews;
        else
          error('Trx file not specified');
        end
      end

      % Handle custom output trk files if provided
      if length(parts) >= partIndex + nviews - 1
        outputTrkFiles = parts(partIndex:partIndex + nviews - 1);
        for i = 1:nviews
          if ~isempty(outputTrkFiles{i})
            % Validate that the directory exists for the output file
            [outputDir,~,~] = fileparts(outputTrkFiles{i});
            if ~isempty(outputDir) && ~exist(outputDir,'dir')
              error('Output directory does not exist: %s',outputDir);
            end
          end
        end
        movdataout.trkfiles = outputTrkFiles;
      end

      % Set other optional fields to defaults
      if obj.lObj.cropProjHasCrops
        movdataout.cropRois = repmat({nan(1,4)},[1,nviews]);
      end

      if nviews > 1
        movdataout.calibrationfiles = [];
      end

      movdataout.targets = [];
      movdataout.f0s = [];
      movdataout.f1s = [];
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
            res = questdlg('Save list of movies to track to resume failed tracking jobs?');
            if strcmpi(res,'Cancel'),
              return;
            elseif strcmpi(res,'Yes'),
              success = obj.loadOrSaveToFile('save');
              if ~success,
                return;
              end
            end
          end

          % Check for existing _detect files and ask user if they want to continue
          userChoice = obj.checkAndPromptForDetectFiles();
          switch userChoice
            case 'cancel'
              return; % User cancelled
            case 'use_detect'
              obj.toTrack.docontinue = true;
            case 'new_tracking'
              obj.toTrack.docontinue = false;
              % Continue with normal tracking
            case 'no_detect_files'
              obj.toTrack.docontinue = false;
              % Continue with normal tracking
          end

          obj.lObj.trackBatch('toTrack',obj.toTrack,'track_type',tag);
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

        % Set defaults if not present in loaded JSON
        if ~isfield(obj.toTrack, 'link_type')
          obj.toTrack.link_type = 'simple';
        end
        if ~isfield(obj.toTrack, 'id_maintain_identity')
          obj.toTrack.id_maintain_identity = false;
        end

        obj.page = 1;
        obj.setNPages();
        obj.updateMovieList();
        obj.updateLinkingControls();
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
      movie = h.Value;

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
        
        % Update original data and restore truncated display
        obj.originalMovieFiles{moviei} = movie;
        if tfgentrk
          obj.originalTrkFiles{moviei} = trk;
        end
        if obj.isma && tfgendetect
          obj.originalDetectFiles{moviei} = detect;
        end
      end
      
    end
    function edit_trx_Callback(obj,h,e,itemi)
      moviei = obj.item2MovieIdx(itemi);
      if moviei>obj.nmovies
        obj.pb_add_Callback([],[],[]);
        return;
      end
      trx = h.Value;
      obj.toTrack.trxfiles{moviei,1} = trx;
      
      % Update original data
      if moviei <= length(obj.originalTrxFiles)
        obj.originalTrxFiles{moviei} = trx;
      end
      
      % mutating .toTrack outside setMovData
      obj.needsSave = true;
    end

    function edit_trk_Callback(obj,h,e,itemi)
      moviei = obj.item2MovieIdx(itemi);
      if moviei>obj.nmovies
        obj.pb_add_Callback([],[],[]);
        return;
      end
      trk = h.Value;
      obj.toTrack.trkfiles{moviei,1} = trk;
      
      % Update original data
      if moviei <= length(obj.originalTrkFiles)
        obj.originalTrkFiles{moviei} = trk;
      end
      
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
      trkpatnew = strtrim(h.Value);
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
            obj.defaultdetectpat = [obj.defaulttrkpat '.tracklet'];
            obj.apply_macro_allmovies();
          case 'No'
            obj.defaulttrkpat = trkpatnew;
            obj.defaultdetectpat = [obj.defaulttrkpat '.tracklet'];
          case 'Cancel'
            % revert
            h.Value = obj.defaulttrkpat;            
        end
      else
        obj.defaulttrkpat = trkpatnew;
        obj.defaultdetectpat = [obj.defaulttrkpat '.tracklet'];

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
      trxpat = strtrim(h.Value);
      if strcmp(trxpat,obj.defaulttrxpat)
        return;
      end
      obj.defaulttrxpat = trxpat;
    end
    
    function figureResizeCallback(obj, src, evt)
      % Called when the figure is resized - re-truncate file paths
      if ~isempty(obj.originalMovieFiles) && isvalid(obj.gdata.fig)
        obj.updateTruncatedFilePathsWithMode();
      end
      % Update radio button positions on resize for multi-animal projects
      if obj.lObj.maIsMA && isfield(obj.gdata, 'bg_linking') && isvalid(obj.gdata.bg_linking)
        obj.updateLinkingButtonPositions();
      end
      % Update path toggle button positions on resize
      if isfield(obj.gdata, 'bg_path') && isvalid(obj.gdata.bg_path)
        obj.updatePathTogglePositions();
      end
    end
    
    function updateLinkingButtonPositions(obj)
      % Calculate and update radio button positions based on button group size
      if ~isfield(obj.gdata, 'bg_linking') || ~isvalid(obj.gdata.bg_linking)
        return;
      end
      
      try
        % Get the button group position in pixels
        bgPos = getpixelposition(obj.gdata.bg_linking);
        bgWidth = bgPos(3);
        bgHeight = bgPos(4);
        
        % Calculate button dimensions with padding
        padding = 10;  % pixels
        buttonHeight = max(20, bgHeight - 2*padding);  % minimum 20px height
        buttonWidth = (bgWidth - 4*padding) / 3;  % divide width by 3 with padding
        
        % Calculate positions for each button
        y = (bgHeight - buttonHeight) / 2;  % center vertically
        
        % Update positions
        obj.gdata.rb_simple.Position = [padding, y, buttonWidth, buttonHeight];
        obj.gdata.rb_motion.Position = [padding + buttonWidth + padding, y, buttonWidth, buttonHeight];
        obj.gdata.rb_identity.Position = [padding + 2*(buttonWidth + padding), y, buttonWidth, buttonHeight];
        
      catch ME
        % Silently handle errors during resize
        warning(ME.identifier,'Error updating linking button positions: %s', ME.message);
      end
    end
    
    function windowButtonDownFcn(obj, src, evt)
      % Figure-level mouse click handler to detect clicks on edit fields
      try
        % Get the current point in the figure
        currentPoint = get(src, 'CurrentPoint');
        
        % Find which UI control was clicked by checking positions
        clickedControl = [];
        fieldType = '';
        itemi = [];
        
        % Check each edit control to see if click was within its bounds
        for i = 1:obj.nmovies_per_page
          if strcmp(get(obj.gdata.edit_movie(i), 'Visible'), 'on')
            if obj.isPointInControl(currentPoint, obj.gdata.edit_movie(i))
              clickedControl = obj.gdata.edit_movie(i);
              fieldType = 'movie';
              itemi = i;
              break;
            end
          end
          
          if strcmp(get(obj.gdata.edit_trk(i), 'Visible'), 'on')
            if obj.isPointInControl(currentPoint, obj.gdata.edit_trk(i))
              clickedControl = obj.gdata.edit_trk(i);
              fieldType = 'trk';
              itemi = i;
              break;
            end
          end
          
        end
        
        % If we found a clicked edit control, expand its path
        if ~isempty(clickedControl) && ~isempty(fieldType) && ~isempty(itemi)
          obj.expandEditFieldPath(clickedControl, itemi, fieldType);
        end
        
      catch ME
        % Silently handle any errors to avoid interrupting UI
        warning(ME.identifier,'Error in windowButtonDownFcn: %s', ME.message);
      end
    end
    
    function tf = isPointInControl(obj, point, control)
      % Check if a point is within a control's bounds
      try
        pos = get(control, 'Position');
        % pos = [x, y, width, height] in normalized coordinates
        tf = point(1) >= pos(1) && point(1) <= pos(1) + pos(3) && ...
             point(2) >= pos(2) && point(2) <= pos(2) + pos(4);
      catch
        tf = false;
      end
    end
    
    function expandEditFieldPath(obj, h, itemi, fieldType)
      % Show full path in edit field if currently truncated
      currentString = get(h, 'Value');
      moviei = obj.item2MovieIdx(itemi);
      
      if moviei <= obj.nmovies
        originalPath = '';
        switch fieldType
          case 'movie'
            if moviei <= length(obj.originalMovieFiles)
              originalPath = obj.originalMovieFiles{moviei};
            end
          case 'trx'
            if moviei <= length(obj.originalTrxFiles)
              originalPath = obj.originalTrxFiles{moviei};
            end
          case 'trk'
            if moviei <= length(obj.originalTrkFiles)
              originalPath = obj.originalTrkFiles{moviei};
            end
          case 'detect'
            if obj.isma && moviei <= length(obj.originalDetectFiles)
              originalPath = obj.originalDetectFiles{moviei};
            end
        end
        
        % If current string is truncated (different from original), show full path
        if ~isempty(originalPath) && ~strcmp(currentString, originalPath)
          % Check if current string looks like truncated version (contains ...)
          if contains(currentString, '...')
            set(h, 'Value', originalPath);
            % Pause briefly to let the user see the change
            pause(0.1);
          end
        end
      end
    end
    
    function expandEditFieldOnChange(obj, h, evt, itemi, fieldType)
      % Called when user starts changing the value - expand truncated paths
      moviei = obj.item2MovieIdx(itemi);
      
      if moviei <= obj.nmovies
        originalPath = '';
        switch fieldType
          case 'movie'
            if moviei <= length(obj.originalMovieFiles)
              originalPath = obj.originalMovieFiles{moviei};
            end
          case 'trx'
            if moviei <= length(obj.originalTrxFiles)
              originalPath = obj.originalTrxFiles{moviei};
            end
          case 'trk'
            if moviei <= length(obj.originalTrkFiles)
              originalPath = obj.originalTrkFiles{moviei};
            end
          case 'detect'
            if obj.isma && moviei <= length(obj.originalDetectFiles)
              originalPath = obj.originalDetectFiles{moviei};
            end
        end
        
        % If current string is truncated (different from original), show full path
        if ~isempty(originalPath) && ~strcmp(evt.Value, originalPath)
          % Check if current string looks like truncated version (contains ...)
          if contains(evt.Value, '...')
            % Set the field to show the original path for editing
            h.Value = originalPath;
          end
        end
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
      obj.updateTruncatedFilePathsWithMode();
    end

    function togglePathDisplayMode(obj)
      % Toggle between showing path ends and path starts in file paths
      obj.showPathEnds = ~obj.showPathEnds;

      % Update toggle button values to reflect current mode
      if isfield(obj.gdata, 'tb_path_starts') && isvalid(obj.gdata.tb_path_starts)
        obj.gdata.tb_path_starts.Value = ~obj.showPathEnds;
      end
      if isfield(obj.gdata, 'tb_path_ends') && isvalid(obj.gdata.tb_path_ends)
        obj.gdata.tb_path_ends.Value = obj.showPathEnds;
      end

      % Update all displayed file paths
      obj.updateTruncatedFilePathsWithMode();
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
    
    function updateTruncatedFilePathsWithMode(obj)
      % Re-truncate file paths based on current display mode preference
      for i = 1:obj.nmovies_per_page
        moviei = obj.item2MovieIdx(i);
        if moviei <= obj.nmovies && moviei <= size(obj.originalMovieFiles, 1)
          % Update movie file path
          if ~isempty(obj.originalMovieFiles{moviei})
            if obj.showPathEnds
              % Show truncated path ends
              displayMovFile = PathTruncationUtils.truncateFilePath(...
                obj.originalMovieFiles{moviei}, 'component', obj.gdata.edit_movie(i), 'startFraction', 0);
            else
              % Show whole path when showing path starts
              displayMovFile = obj.originalMovieFiles{moviei};
            end
            set(obj.gdata.edit_movie(i), 'Value', displayMovFile);
          end
          
          % Update trk file path
          if moviei <= size(obj.originalTrkFiles, 1) && ~isempty(obj.originalTrkFiles{moviei})
            if obj.showPathEnds
              % Show truncated path ends
              displayTrkFile = PathTruncationUtils.truncateFilePath(...
                obj.originalTrkFiles{moviei}, 'component', obj.gdata.edit_trk(i), 'startFraction', 0);
            else
              % Show whole path when showing path starts
              displayTrkFile = obj.originalTrkFiles{moviei};
            end
            set(obj.gdata.edit_trk(i), 'Value', displayTrkFile);
          end
          if obj.hasTrx && moviei <= size(obj.originalTrxFiles, 1) && ~isempty(obj.originalTrxFiles{moviei})
            if obj.showPathEnds
              % Show truncated path ends
              displayTrxFile = PathTruncationUtils.truncateFilePath(...
                obj.originalTrxFiles{moviei}, 'component', obj.gdata.edit_trk(i), 'startFraction', 0);
            else
              % Show whole path when showing path starts
              displayTrxFile = obj.originalTrxFiles{moviei};
            end
            set(obj.gdata.edit_trx(i), 'Value', displayTrxFile);
          end
        end
      end
    end
    
    function linkingTypeChanged(obj, src, evt)
      % Callback for linking type radio button group
      selectedButton = evt.NewValue;

      % Determine which linking type was selected
      if selectedButton == obj.gdata.rb_simple
        obj.toTrack.link_type = 'simple';
      elseif selectedButton == obj.gdata.rb_motion
        obj.toTrack.link_type = 'motion';
      elseif selectedButton == obj.gdata.rb_identity
        obj.toTrack.link_type = 'identity';
      end

      obj.needsSave = true;

      % Enable/disable maintain identities checkbox based on identity linking
      if isfield(obj.gdata, 'chk_maintain_identities') && isvalid(obj.gdata.chk_maintain_identities)
        isIdentityLinking = strcmp(obj.toTrack.link_type, 'identity');
        obj.gdata.chk_maintain_identities.Enable = isIdentityLinking;

        % If not identity linking, uncheck maintain identities
        if ~isIdentityLinking
          obj.gdata.chk_maintain_identities.Value = false;
          obj.toTrack.id_maintain_identity = false;
        end
      end
    end
    
    function idMaintainIdentityChanged(obj, h, evt)
      % Callback for maintain identities checkbox
      obj.toTrack.id_maintain_identity = h.Value;
      obj.needsSave = true;
    end

    function updateLinkingControls(obj)
      % Update radio buttons and checkbox to reflect loaded data
      if ~isfield(obj.gdata, 'rb_simple') || ~isvalid(obj.gdata.rb_simple)
        return; % GUI not created yet
      end

      % Update radio button values
      obj.gdata.rb_simple.Value = strcmp(obj.toTrack.link_type, 'simple');
      obj.gdata.rb_motion.Value = strcmp(obj.toTrack.link_type, 'motion');
      obj.gdata.rb_identity.Value = strcmp(obj.toTrack.link_type, 'identity');

      % Update maintain identities checkbox
      if isfield(obj.gdata, 'chk_maintain_identities') && isvalid(obj.gdata.chk_maintain_identities)
        obj.gdata.chk_maintain_identities.Value = obj.toTrack.id_maintain_identity;
        obj.gdata.chk_maintain_identities.Enable = strcmp(obj.toTrack.link_type, 'identity');
      end
    end

    function userChoice = checkAndPromptForDetectFiles(obj)
      % Check for existing _detect files and prompt user if they want to continue with them
      % Returns user choice: 'cancel', 'use_detect', 'new_tracking', or 'no_detect_files'

      % Get all trk files and check for corresponding _detect files
      trkfiles = obj.toTrack.trkfiles;
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

      % If no _detect files exist, proceed normally
      if isempty(existingDetectFiles)
        userChoice = 'no_detect_files';
        return;
      end

      % Create table data for display (3 columns: file, timestamp, status)
      tableData = cell(length(existingDetectFiles), 3);
      for i = 1:length(existingDetectFiles)
        tableData{i, 1} = existingDetectFiles{i};
        tableData{i, 2} = detectTimestamps{i};
        tableData{i, 3} = predictionStatus{i};
      end

      % Show dialog with table
      userChoice = obj.showDetectFilesDialog(tableData);

      % Return user choice without modifying toTrack here
      if isempty(userChoice)
        userChoice = 'cancel'; % Default if dialog was closed
      end
    end

    function choice = showDetectFilesDialog(obj, tableData)
      % Show dialog with table of existing detect files
      choice = '';

      % Create dialog
      d = dialog('Position', [300, 300, 800, 400], 'Name', 'Existing Detection Files Found');

      % Title
      uicontrol('Parent', d, 'Style', 'text', ...
                'Position', [20, 350, 760, 30], ...
                'String', 'Existing detection files (_detect.trk) were found. Would you like to continue tracking using these files?', ...
                'FontWeight', 'bold', 'FontSize', 12);

      % Table
      columnNames = {'Detection File', 'Last Modified', 'Prediction Status'};
      uit = uitable('Parent', d, 'Data', tableData, ...
                    'ColumnName', columnNames, ...
                    'Position', [20, 100, 760, 240], ...
                    'ColumnEditable', [false, false, false]);

      % Info text
      uicontrol('Parent', d, 'Style', 'text', ...
                'Position', [20, 70, 760, 20], ...
                'String', 'Choose "Use Detect Files" to continue with existing detections, or "Start New Tracking" to overwrite them.');

      % Buttons
      uicontrol('Parent', d, 'Position', [200, 20, 100, 30], ...
                'String', 'Cancel', ...
                'Callback', @(btn, event) setChoiceAndClose('cancel'));

      uicontrol('Parent', d, 'Position', [320, 20, 120, 30], ...
                'String', 'Use Detect Files', ...
                'Callback', @(btn, event) setChoiceAndClose('use_detect'));

      uicontrol('Parent', d, 'Position', [460, 20, 120, 30], ...
                'String', 'Start New Tracking', ...
                'Callback', @(btn, event) setChoiceAndClose('new_tracking'));

      centerOnParentFigure(d,obj.gdata.fig);
      % Wait for user choice
      uiwait(d);

      function setChoiceAndClose(selectedChoice)
        choice = selectedChoice;
        if isvalid(d)
          delete(d);
        end
      end
    end

  end
end

