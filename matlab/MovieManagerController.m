classdef MovieManagerController < handle
  properties (SetAccess=private)
    hFig % scalar handle to MovieManager fig
    
    labeler % scalar labeler Obj
    listeners % cell array of listener objs
    
    tblMovies
    tblMovieSet
    tabHandles % [2] "handles" struct array
    
    % Store original (untruncated) data for dynamic resizing
    originalMovNames
    originalTrxNames
    originalMovsHaveLbls
    
    % User preference for path truncation (true = show ends, false = show starts)
    showPathEnds = true
  end
  properties (Constant)
    JTABLEPROPS_NOTRX = {'ColumnName',{'Movie' 'Num Labels'},...
                         'ColumnWidth',{'1x',100}};
    JTABLEPROPS_TRX = {'ColumnName',{'Movie' 'Trx' 'Num Labels'},...
                       'ColumnWidth',{'2x','1x',100}};
  end
  
  events
    tableClicked % fires when any movietable is clicked
  end
  
  methods
    
    % MovieManagerController messages between Labeler and Tables
    % 1. Labeler/clients can fetch current selection in Table
    % 2. Labeler prop changes fire MMC listeners to update Table content
    % 3. MMC Tables can set current movie in Labeler based on user action
    % 4. MMC buttons can add/rm labeler movies

    function obj = MovieManagerController(lObj)
      assert(isa(lObj,'Labeler'));
      %obj.hFig = MovieManager(obj);
      obj.labeler = lObj;
      
      obj.hFig = uifigure('Units','pixels','Position',[951,1400,733,436],...
        'Name','Manage Movies',...
        'AutoResizeChildren','off',...
        'SizeChangedFcn',@(src,evt) obj.figureResizeCallback(src,evt));
      handles.figure1 = obj.hFig;
      %obj.hFig.CloseRequestFcn = @(hObject,eventdata) obj.CloseRequestFcn(hObject,eventdata);

      handles.gl = uigridlayout(obj.hFig,[4,1],'RowHeight',obj.getGridLayoutRowHeights(),'tag','gl');

      handles.tblMovies = uitable(handles.gl,...
        'ColumnName',{'Movie','Has Lbls'},...
        'ColumnWidth',{'1x',100},...
        'tag','tblMovies',...
        'CellSelectionCallback',@(src,evt) obj.cellSelectionCallbackTblMovies(src,evt),...
        'DoubleClickedFcn',@(src,evt) obj.doubleClickFcnCallbackTblMovies(src,evt));
      
      obj.tblMovies = handles.tblMovies;
      handles.labelSet = uilabel('Parent',handles.gl,'Text','',...
        'Visible',onIff(lObj.nview > 1),'HorizontalAlignment','center');

      rownames = arrayfun(@(x) sprintf('View %d',x), 1:lObj.nview,'Uni',0);
      handles.tblMovieSet = uitable(handles.gl,...
        'ColumnName',{},'tag','tblMovieSet',...
        'RowName',rownames,'Visible',onIff(lObj.nview > 1));

      obj.tblMovieSet = handles.tblMovieSet;

      handles.gl_buttons = uigridlayout(handles.gl,[1,5],'Padding',[0,0,0,0],'tag','gl_buttons');

      % Create button group for path display toggle buttons
      handles.bg_path = uibuttongroup(handles.gl_buttons,...
        'BackgroundColor',[0.94 0.94 0.94],...
        'BorderType','none',...
        'SelectionChangedFcn',@(src,evt) obj.pathToggleChanged(src,evt));

      % Load icon images
      leftAlignIcon = imread(fullfile(fileparts(mfilename('fullpath')), 'util', 'align_left.png'));
      rightAlignIcon = imread(fullfile(fileparts(mfilename('fullpath')), 'util', 'align_right.png'));
      if ndims(leftAlignIcon)==2
        leftAlignIcon = repmat(leftAlignIcon,[1 1 3]);
      end
      if ndims(rightAlignIcon)==2
        rightAlignIcon = repmat(rightAlignIcon,[1 1 3]);
      end

      % "Starts" toggle button (left side)
      handles.tb_path_starts = uitogglebutton(handles.bg_path,...
        'Text','',...
        'Icon',leftAlignIcon,...
        'Tooltip','Show path starts',...
        'FontColor','k','BackgroundColor',[1,1,1],...
        'FontWeight','bold',...
        'Value',~obj.showPathEnds,...
        'Tag','togglebutton_path_starts');

      % "Ends" toggle button (right side)
      handles.tb_path_ends = uitogglebutton(handles.bg_path,...
        'Text','',...
        'Icon',rightAlignIcon,...
        'Tooltip','Show path ends',...
        'FontColor','k','BackgroundColor',[1,1,1],...
        'FontWeight','bold',...
        'Value',obj.showPathEnds,...
        'Tag','togglebutton_path_ends');

      if lObj.gtIsGTMode,
        handles.pbSwitch = uibutton(handles.gl_buttons,'Text','GT Frames','tag','pbGTFrames',...
          'ButtonPushedFcn',@(src,evt) cbkPushButton(obj,src,evt));
      else
        handles.pbSwitch = uibutton(handles.gl_buttons,'Text','Switch to Movie','tag','pbSwitch',...
          'ButtonPushedFcn',@(src,evt) cbkPushButton(obj,src,evt));
      end
      handles.pbNextUnlabeled = uibutton(handles.gl_buttons,'Text','Next Unlabeled','tag','pbNextUnlabeled',...
        'ButtonPushedFcn',@(src,evt) cbkPushButton(obj,src,evt));
      if lObj.gtIsGTMode,
        handles.pbNextUnlabeled.Visible = 'off';
      end

      handles.pbAdd = uibutton(handles.gl_buttons,'Text','Add Movie','tag','pbAdd',...
        'ButtonPushedFcn',@(src,evt) cbkPushButton(obj,src,evt));
      handles.pbRm = uibutton(handles.gl_buttons,'Text','Remove Movie','tag','pbRm',...
        'ButtonPushedFcn',@(src,evt) cbkPushButton(obj,src,evt));


      handles.menu_file = uimenu('Tag','menu_file','Text','File','Parent',obj.hFig);
      handles.menu_file_add_movies_from_text_file = uimenu('Tag','menu_file_add_movies_from_text_file',...
        'Text','Add movies from text file','Parent',handles.menu_file);

      handles.menu_file_add_movies_from_text_file.MenuSelectedFcn = ...
          @(s,e)obj.mnuFileAddMoviesBatch();

      guidata(obj.hFig,handles);

      set(obj.hFig,'MenuBar','None');
      obj.update();
      obj.hFig.Visible = 'on';
      
      %PROPS = {};
      %GTPROPS = {};
      lObjs = cell(0,1);
      lObjs{end+1,1} = addlistener(lObj,'didSetMovieFilesAll',@(s,e)(obj.lblerLstnCbkUpdateTable(s,e)));
      lObjs{end+1,1} = addlistener(lObj,'didSetMovieFilesAllHaveLbls',@(s,e)(obj.lblerLstnCbkUpdateTable(s,e)));      
      lObjs{end+1,1} = addlistener(lObj,'didSetTrxFilesAll',@(s,e)(obj.lblerLstnCbkUpdateTable(s,e)));      
      %lObjs{end+1,1} = listenPropsPostSet(lObj,PROPS,@(s,e)obj.lblerLstnCbkUpdateTable(s,e));

      lObjs{end+1,1} = addlistener(lObj,'didSetMovieFilesAllGT',@(s,e)(obj.lblerLstnCbkUpdateTableGT(s,e)));
      lObjs{end+1,1} = addlistener(lObj,'didSetMovieFilesAllGTHaveLbls',@(s,e)(obj.lblerLstnCbkUpdateTableGT(s,e)));      
      lObjs{end+1,1} = addlistener(lObj,'didSetTrxFilesAllGT',@(s,e)(obj.lblerLstnCbkUpdateTableGT(s,e)));      
      %lObjs{end+1,1} = listenPropsPostSet(lObj,GTPROPS,@(s,e)obj.lblerLstnCbkUpdateTableGT(s,e));

      lObjs{end+1,1} = addlistener(lObj,'didLoadProject',@(s,e)obj.lblerLstnCbkProjLoaded(s,e));
      lObjs{end+1,1} = addlistener(lObj,'newMovie',@(s,e)obj.lblerLstnCbkNewMovie(s,e));
      lObjs{end+1,1} = addlistener(lObj,'gtIsGTModeChanged',@(s,e)obj.lblerLstnCbkGTMode(s,e));

      obj.listeners = lObjs;
      obj.hFig.DeleteFcn = @obj.lclDeleteFig;
      
      centerfig(obj.hFig,obj.labeler.gdata.mainFigure_);
      getframe(obj.hFig);

      % Initialize button positions and update data
      obj.updateToggleButtonPositions();
      obj.figureResizeCallback(obj.hFig,[]);
    end

    function lclDeleteFig(obj,src,evt)
      listenObjs = obj.listeners;
      for i=1:numel(listenObjs)
        o = listenObjs{i};
        if isvalid(o)
          delete(o);
        end
      end
    end
    
    function rowheights = getGridLayoutRowHeights(obj)
      lObj = obj.labeler;
      if lObj.nview == 1,
        rowheights = {'1x',0,0,40};
      else
        rowheights = {'1x',20,20*(lObj.nview+1),40};
      end
    end

    function cellSelectionCallbackTblMovies(obj,src,evt)
      rows = evt.Indices(:,1);
      if obj.labeler.nview > 1,
        gdata = guidata(obj.hFig);
        if numel(rows) ~= 1,
          obj.tblMovieSet.Data = cell(0,1);
          gdata.labelSet.Text = '';
          return;
        end
        % Get movie set data and apply path truncation
        movieSetData = obj.labeler.movieFilesAllGTaware(rows,:)';
        obj.tblMovieSet.Data = obj.truncateMovieSetPaths(movieSetData);
        gdata.labelSet.Text = sprintf('Selected set %d',rows);
      end      
      obj.notify('tableClicked');
      % iMov = obj.getSelectedMoviesTblMovies();
      % if ~isempty(iMov)
      %   iMov = iMov(1);
      %   obj.tblCbkMovieSelected(iMov);
      % end
    end

    function doubleClickFcnCallbackTblMovies(obj,src,evt)
      row = evt.InteractionInformation.DisplayRow;
      if isempty(row),
        return;
      end
      obj.tblCbkMovieSelected(row);
    end

    function delete(obj)
      delete(obj.hFig);
      for i=1:numel(obj.listeners)
        delete(obj.listeners{i});
      end
      obj.listeners = [];      
    end
   
        
  end
  
  methods
    
    function setVisible(obj,tf)
      obj.hFig.Visible = onIff(tf);
      if tf
        figure(obj.hFig);
      end
    end

    function tf = isValid(obj)
      tf = isvalid(obj.hFig);
    end

    function idx = getSelectedMovies(obj)
      idx = unique(obj.tblMovies.Selection(:,1),'stable');
    end
    
    function tblCbkMovieSelected(obj,iMov)
      assert(isscalar(iMov) && iMov>0);
      % iMov is gt-aware movie index (unsigned)
      obj.labeler.movieSetGUI(iMov);
    end
    
    function cbkPushButton(obj,src,~)
      lObj = obj.labeler;
      
      switch src.Tag
        case 'pbAdd'
          obj.addLabelerMovie(); % can throw
        case 'pbRm'
          obj.rmLabelerMovie();
        case 'pbSwitch' 
          iMov = obj.getSelectedMovies();
          if ~isempty(iMov)
            iMov = iMov(1);
            obj.tblCbkMovieSelected(iMov);
          end
        case 'pbNextUnlabeled'
          iMov = find(~lObj.movieFilesAllHaveLbls,1);
          if isempty(iMov)
            msgbox('All movies are labeled!');
          else
            lObj.movieSetGUI(iMov);
          end
        case 'pbGTFrames'
          lObj.gtShowGTManager();
        otherwise
          assert(false);
      end
    end   
  
    function lblerLstnCbkUpdateTable(obj,~,~)
      if ~obj.labeler.gtIsGTMode,
        obj.hlpLblerLstnCbkUpdateTable();
      end
    end
    
    function lblerLstnCbkUpdateTableGT(obj,~,~)
      if obj.labeler.gtIsGTMode,
        obj.hlpLblerLstnCbkUpdateTable();
      end
    end
    
    function lblerLstnCbkProjLoaded(obj,~,~)
      obj.hlpLblerLstnCbkUpdateTable();
    end
    
    function lblerLstnCbkNewMovie(obj,~,~)
      obj.updateMMTblRowSelection();
    end
    
    function lblerLstnCbkGTMode(obj,~,~)
      obj.update();
    end
    
    function mnuFileAddMoviesBatch(obj)
      lObj = obj.labeler;

      lastTxtFile = lObj.rcGetProp('lastMovieBatchFile');
      if ~isempty(lastTxtFile)
        [~,~,ext] = fileparts(lastTxtFile);
        ext = ['*' ext];
        file0 = lastTxtFile;
      else
        ext = '*.txt';
        file0 = pwd;
      end
      [fname,pname] = uigetfile(ext,'Select movie batch file',file0);
      if isequal(fname,0)
        return;
      end
      
      nmovieOrig = lObj.nmoviesGTaware;
      fname = fullfile(pname,fname);
      lObj.movieAddBatchFile(fname);
      lObj.rcSaveProp('lastMovieBatchFile',fname);
      if nmovieOrig==0 && lObj.nmoviesGTaware>0
        lObj.movieSetGUI(1);
      end
    end

    function bringWindowToFront(obj)
      obj.setVisible(true) ;  % make sure is visible
      figure(obj.hFig) ;
    end
    
    function updatePointer(obj)
      % Update the mouse pointer to reflect the Labeler state.
      is_busy = obj.labeler.isStatusBusy ;
      pointer = fif(is_busy, 'watch', 'arrow') ;
      set(obj.hFig, 'Pointer', pointer) ;
    end  % function    
  end  % methods
  
  methods (Hidden)


    function update(obj)

      lObj = obj.labeler;
      if lObj.isinit
        return;
      end
      tfGT = lObj.gtIsGTMode;
      assert(islogical(tfGT));
      
      obj.updateMovieData();
      obj.updatePushButtonsEnable();
      obj.updateMMTblRowSelection();
      obj.updateMenusEnable();

    end
    
    function updatePushButtonsEnable(obj)

      handles = guidata(obj.hFig);
      lObj = obj.labeler;
      if lObj.gtIsGTMode,
        set(handles.pbSwitch,'Text','GT Frames','tag','pbGTFrames');
        handles.pbNextUnlabeled.Visible = 'off';
      else
        set(handles.pbSwitch,'Text','Switch to Movie','tag','pbSwitch');
        handles.pbNextUnlabeled.Visible = 'on';
      end
    end
    
    function updateMenusEnable(obj)
      gdata = guidata(obj.hFig);
      gdata.menu_file_add_movies_from_text_file.Enable = 'on';
    end
    
    function setSelectedMovie(obj,iMov)
      if ~obj.labeler.isinit,
        return;
      end
      if isempty(obj.tblMovies.Data),
        return;
      end
      if isempty(iMov) || iMov == 0,
        obj.tblMovies.Selection = zeros(0,2);
      elseif isequal(obj.tblMovies.Selection(1),iMov),
        return;
      else
        obj.tblMovies.Selection = [iMov,1];
      end
      % todo update movieset table
    end

    function updateMMTblRowSelection(obj)
      % Update one of the MM tables per lObj.currMovie, lObj.gtIsGTMode
      
      iMov = obj.labeler.currMovie;
      obj.setSelectedMovie(iMov);

    end

    function updateMovieData(obj,movNames,trxNames,movsHaveLbls)

      lObj = obj.labeler;
      tfGT = lObj.gtIsGTMode ;
      gdata = guidata(obj.hFig);

      if nargin < 2,

        PROPS = Labeler.gtGetSharedPropsStc(tfGT);
        movNames = lObj.(PROPS.MFA);
        trxNames = lObj.(PROPS.TFA);
        movsHaveLbls = lObj.(PROPS.MFAHL);

      end

      if ~isequal(size(movNames,1),size(trxNames,1),numel(movsHaveLbls))
        % intermediate state, take no action
        return
      end
      
      gdata.gl.RowHeight = obj.getGridLayoutRowHeights();
      obj.tblMovieSet.Visible = onIff(lObj.nview > 1);
      gdata.labelSet.Visible = onIff(lObj.nview > 1);

      % Store original data for dynamic truncation
      obj.originalMovNames = movNames;
      obj.originalTrxNames = trxNames;
      obj.originalMovsHaveLbls = movsHaveLbls;
      
      % Update table with truncated data
      obj.updateTruncatedTableData();

    end

    function hlpLblerLstnCbkUpdateTable(obj)
      lObj = obj.labeler;
      if lObj.isinit
        return
      end

      if ~lObj.hasProject
        return
        % error('MovieManagerController:proj',...
        %   'Please open/create a project first.');
      end

      obj.updateMovieData();
      obj.updateMMTblRowSelection();
    end
    
    function addLabelerMovie(obj)
      lObj = obj.labeler;
      nmovieOrig = lObj.nmoviesGTaware;
      if lObj.nview==1
        [tfsucc,movfile,trxfile] = promptGetMovTrxFiles(true,lObj.projectHasTrx);
        if ~tfsucc
          return;
        end
        try
          lObj.movieAdd(movfile,trxfile);
        catch ME,
          uiwait(errordlg(getReport(ME,'basic','hyperlinks','off'),'Error adding movies'));
          return;
        end

      else
        assert(lObj.nTargets==1,'Adding trx files currently unsupported.');
        lastmov = lObj.rcGetProp('lbl_lastmovie');
        if isempty(lastmov)
          lastmovpath = pwd;
        else
          lastmovpath = fileparts(lastmov);
        end
        movfiles = uipickfiles(...
          'Prompt','Select movie set',...
          'FilterSpec',lastmovpath,...
          'NumFiles',lObj.nview);
        if isequal(movfiles,0)
          return;
        end
        try
          lObj.movieSetAdd(movfiles);
        catch ME,
          uiwait(errordlg(getReport(ME,'basic','hyperlinks','off'),'Error adding movies'));
          return;
        end
      end
      if nmovieOrig==0 && lObj.nmoviesGTaware>0
        lObj.movieSetGUI(1,'isFirstMovie',true);
      end
    end
    
    function rmLabelerMovie(obj)
      selRow = obj.getSelectedMovies();
      selRow = sort(selRow);
      n = numel(selRow);
      lObj = obj.labeler;
      for i = n:-1:1
        row = selRow(i);
        try
          tfSucc = lObj.movieRmGUI(row);
        catch ME,
          uiwait(errordlg(getReport(ME,'basic','hyperlinks','off'),'Error removing movie'));
          break;
        end
        if ~tfSucc
          % user stopped/canceled
          break;
        end
      end
    end
    
    function figureResizeCallback(obj, src, evt)
      % Called when the figure is resized - re-truncate table data and update button positions
      if ~isempty(obj.originalMovNames)
        obj.updateTruncatedTableData();
      end

      % Update toggle button positions within the button group
      obj.updateToggleButtonPositions();

      % Update movieset table if it has data
      obj.updateMovieSetTable();
    end

    function updateToggleButtonPositions(obj)
      % Update the positions of toggle buttons within the button group
      try
        handles = guidata(obj.hFig);

        % Get button group position
        bgPos = handles.bg_path.Position;
        buttonWidth = (bgPos(3) - 4) / 2; % Two buttons with 2px spacing each
        buttonHeight = bgPos(4) - 4; % Leave 2px padding top/bottom

        % Position "Starts" button (left side)
        if isfield(handles, 'tb_path_starts') && isvalid(handles.tb_path_starts)
          handles.tb_path_starts.Position = [2, 2, buttonWidth, buttonHeight];
        end

        % Position "Ends" button (right side)
        if isfield(handles, 'tb_path_ends') && isvalid(handles.tb_path_ends)
          handles.tb_path_ends.Position = [buttonWidth + 4, 2, buttonWidth, buttonHeight];
        end

      catch
        % Silently handle any positioning errors
      end
    end

    function truncatedData = truncateMovieSetPaths(obj, movieSetData)
      % Apply path truncation to movie set table data based on showPathEnds setting
      if isempty(movieSetData)
        truncatedData = movieSetData;
        return;
      end

      try
        % Calculate appropriate column width for movieset table
        tablePos = obj.tblMovieSet.Position;
        tableWidthPixels = tablePos(3) - 20; % Account for margins

        % Get font size from the table component
        try
          origFontUnits = get(obj.tblMovieSet, 'FontUnits');
          set(obj.tblMovieSet, 'FontUnits', 'pixels');
          fontSize = get(obj.tblMovieSet, 'FontSize');
          set(obj.tblMovieSet, 'FontUnits', origFontUnits);
        catch
          fontSize = 14; % Default if getting font size fails
        end

        if obj.showPathEnds
          % Truncate showing path ends
          truncatedData = cellfun(@(x) PathTruncationUtils.truncateFilePath(x, ...
            'maxLength', PathTruncationUtils.calculateMaxCharsForFieldWidth(tableWidthPixels, fontSize), 'startFraction', 0), ...
            movieSetData, 'UniformOutput', false);
        else
          % Show full paths when showing starts
          truncatedData = movieSetData;
        end
      catch
        % If truncation fails, return original data
        truncatedData = movieSetData;
      end
    end

    function columnWidth = calculateMovieColumnWidth(obj)
      % Calculate the actual pixel width of the movie name column
      try
        % Get table position in pixels
        tablePos = obj.tblMovies.Position;
        tableWidthPixels = tablePos(3);
        
        % Get current column widths from table properties
        lObj = obj.labeler;
        tfTrx = lObj.hasTrx && any(cellfun(@(x)~isempty(x), obj.originalTrxNames(:)));
        
        if tfTrx
          % For TRX table: Movie=2x, Trx=1x, NumLabels=100px
          % Total flexible units = 2x + 1x = 3x
          % Available width = tableWidth - 100px (for NumLabels)
          availableWidth = tableWidthPixels - 100-20;
          flexibleUnitWidth = availableWidth / 3;
          columnWidth = 2 * flexibleUnitWidth; % Movie column gets 2x
        else
          % For NOTRX table: Movie=1x, NumLabels=100px
          % Available width = tableWidth - 100px (for NumLabels)
          availableWidth = tableWidthPixels - 100-20;
          columnWidth = availableWidth; % Movie column gets all remaining space
        end
        
        % Ensure minimum width
        %columnWidth = max(columnWidth, 100);
        
      catch
        % Fallback to default if calculation fails
        columnWidth = 300;
      end
    end
    
    function updateTruncatedTableData(obj)
      % Re-calculate truncated data and update table
      if isempty(obj.originalMovNames)
        return;
      end
      
      lObj = obj.labeler;
      tfTrx = lObj.hasTrx && any(cellfun(@(x)~isempty(x), obj.originalTrxNames(:)));
      
      if obj.showPathEnds
        % Calculate current column width and truncate
        columnWidthPixels = obj.calculateMovieColumnWidth();
        
        % Truncate movie names based on current column width
        movSetNames = cellfun(@(x) PathTruncationUtils.truncateFilePath(x, ...
          'maxLength', PathTruncationUtils.calculateMaxCharsForFieldWidth(columnWidthPixels, 12),'startFraction',0), ...
          obj.originalMovNames(:,1), 'UniformOutput', false);
        
        % Truncate trx names based on current column width (if applicable)
        if tfTrx
          % TRX column gets 1x width (1/3 of flexible space)
          trxColumnWidth = obj.calculateMovieColumnWidth() / 2; % Half of movie column width
          trxSetNames = cellfun(@(x) PathTruncationUtils.truncateFilePath(x, ...
            'maxLength', PathTruncationUtils.calculateMaxCharsForFieldWidth(trxColumnWidth, 12),'startFraction',0), ...
            obj.originalTrxNames(:,1), 'UniformOutput', false);
        else
          trxSetNames = obj.originalTrxNames(:,1);
        end
      else
        % Show full paths - no truncation
        movSetNames = obj.originalMovNames(:,1);
        trxSetNames = obj.originalTrxNames(:,1);
      end
      
      % Update table data
      if tfTrx
        dat = [movSetNames trxSetNames num2cell(int64(obj.originalMovsHaveLbls))];
        args = MovieManagerController.JTABLEPROPS_TRX;
      else
        dat = [movSetNames num2cell(int64(obj.originalMovsHaveLbls))];
        args = MovieManagerController.JTABLEPROPS_NOTRX;
      end
      set(obj.tblMovies,args{:},'Data',dat);
      
      % Right-align the movie column(s)
    end
    
    function pathToggleChanged(obj, src, evt)
      % Handle toggle button group selection change for path display
      if strcmp(evt.NewValue.Tag, 'togglebutton_path_ends')
        obj.showPathEnds = true;
      else
        obj.showPathEnds = false;
      end

      % Update the main table display
      obj.updateTruncatedTableData();

      % Update movieset table if it has data
      obj.updateMovieSetTable();
    end

    function updateMovieSetTable(obj)
      % Update movieset table with current path truncation setting
      if obj.labeler.nview > 1 && ~isempty(obj.tblMovieSet) && ~isempty(obj.tblMovieSet.Data)
        % Get currently selected row from main table
        selectedRows = obj.getSelectedMovies();
        if ~isempty(selectedRows)
          rows = selectedRows(1);
          movieSetData = obj.labeler.movieFilesAllGTaware(rows,:)';
          obj.tblMovieSet.Data = obj.truncateMovieSetPaths(movieSetData);
        end
      end
    end

  end  % methods (Hidden)
  
end  % classdef