classdef MovieManagerController < handle
  properties (SetAccess=private)
    parent_ % scalar LabelerController that created this object
    hFig % scalar handle to MovieManager fig

    labeler % scalar labeler Obj
    listeners % cell array of listener objs

    tblMain  % the main table
    tblView  % A table used in multiview projects, showing the movies for the selected movieset

    % UI handles previously kept in guidata(hFig).
    gl  % top-level uigridlayout
    labelSet  % uilabel showing which movieset is selected
    glButtons  % uigridlayout for the bottom button row
    pbSwitch  % uibutton: "Switch to Movie" or "GT Frames"
    pbNextUnlabeled  % uibutton: "Next Unlabeled"
    pbAdd  % uibutton: "Add Movie"
    pbRm  % uibutton: "Remove Movie"
    pbChangePath  % uibutton: "Change Path..."
    menuFile  % uimenu: "File"
    menuFileAddMoviesFromTextFile  % uimenu under File
  end

  methods    
    % MovieManagerController messages between Labeler and Tables
    % 1. Labeler/clients can fetch current selection in Table
    % 2. Labeler prop changes fire MMC listeners to update Table content
    % 3. MMC Tables can set current movie in Labeler based on user action
    % 4. MMC buttons can add/rm labeler movies

    function obj = MovieManagerController(labelerController, labeler)
      % Construct the MMC: build the UI and wire up Labeler listeners.
      assert(isa(labelerController, 'LabelerController'));
      assert(isa(labeler, 'Labeler'));
      obj.parent_ = labelerController;
      lObj = labeler;
      %obj.hFig = MovieManager(obj);
      obj.labeler = lObj;
      
      obj.hFig = uifigure('Units','pixels', ...
                          'Position',[951 1400 733 436], ...
                          'Name','Manage Movies') ;
      %obj.hFig.CloseRequestFcn = @(hObject,eventdata) obj.CloseRequestFcn(hObject,eventdata);

      obj.gl = uigridlayout(obj.hFig,[4,1],'RowHeight',obj.getGridLayoutRowHeights_(),'tag','gl');

      obj.tblMain = uitable(obj.gl,...
        'ColumnName',{'Movie','Has Lbls'},...
        'ColumnWidth',{'1x',70},...
        'Tag','tblMain',...
        'CellSelectionCallback',@(src,evt) obj.selectionChangedTblMovies(src,evt),...
        'DoubleClickedFcn',@(src,evt) obj.doubleClickFcnCallbackTblMovies(src,evt));
      obj.labelSet = ...
        uilabel('Parent',obj.gl,...
                'Text','<no movieset selected>',...
                'Visible',onIff(lObj.nview > 1),...
                'HorizontalAlignment','center');

      rownames = arrayfun(@(x) sprintf('View %d',x), 1:lObj.nview,'Uni',0);
      obj.tblView = uitable(obj.gl,...
        'ColumnName',{},'Tag','tblView',...
        'RowName',rownames,'Visible',onIff(lObj.nview > 1),...
        'CellSelectionCallback',@(src,evt) obj.selectionChangedTblMovieSet(src,evt));

      obj.glButtons = uigridlayout(obj.gl,[1,5],'Padding',[0,0,0,0],'tag','gl_buttons');

      obj.pbSwitch = uibutton(obj.glButtons,...
                              'Tag', 'pbSwitch', ...
                              'ButtonPushedFcn',@(src,evt) cbkPushButton(obj,src,evt));
      obj.pbNextUnlabeled = uibutton(obj.glButtons,'Text','Next Unlabeled','Tag','pbNextUnlabeled',...
        'ButtonPushedFcn',@(src,evt) cbkPushButton(obj,src,evt));
      if lObj.gtIsGTMode,
        obj.pbNextUnlabeled.Visible = 'off';
      end

      obj.pbAdd = uibutton(obj.glButtons,'Text','Add Movie...','tag','pbAdd',...
        'ButtonPushedFcn',@(src,evt) cbkPushButton(obj,src,evt));
      obj.pbRm = uibutton(obj.glButtons,'Text','Remove Movie','tag','pbRm',...
        'ButtonPushedFcn',@(src,evt) cbkPushButton(obj,src,evt));
      obj.pbChangePath = uibutton(obj.glButtons,'Text','Change Path...','tag','pbChangePath',...
        'ButtonPushedFcn',@(src,evt) cbkPushButton(obj,src,evt));

      obj.menuFile = uimenu('Tag','menu_file','Text','File','Parent',obj.hFig);
      obj.menuFileAddMoviesFromTextFile = uimenu('Tag','menu_file_add_movies_from_text_file',...
        'Text','Add movies from text file','Parent',obj.menuFile);

      obj.menuFileAddMoviesFromTextFile.MenuSelectedFcn = ...
          @(s,e)obj.mnuFileAddMoviesBatch();

      set(obj.hFig,'MenuBar','None');
      obj.update();
      
      listenerObjs = cell(0,1);
      listenerObjs{end+1,1} = addlistener(lObj,'didSetMovieFilesAll',@(s,e)(obj.update()));
      listenerObjs{end+1,1} = addlistener(lObj,'didSetMovieFilesAllHaveLbls',@(s,e)(obj.update()));
      listenerObjs{end+1,1} = addlistener(lObj,'didSetTrxFilesAll',@(s,e)(obj.update()));
      listenerObjs{end+1,1} = addlistener(lObj,'didSetMovieFilesAllGT',@(s,e)(obj.update()));
      listenerObjs{end+1,1} = addlistener(lObj,'didSetMovieFilesAllGTHaveLbls',@(s,e)(obj.update()));
      listenerObjs{end+1,1} = addlistener(lObj,'didSetTrxFilesAllGT',@(s,e)(obj.update()));
      listenerObjs{end+1,1} = addlistener(lObj,'didLoadProject',@(s,e)(obj.update()));
      listenerObjs{end+1,1} = addlistener(lObj,'gtIsGTModeChanged',@(s,e)(obj.update()));
      listenerObjs{end+1,1} = addlistener(lObj,'didSetMoviesSelected',...
                                          @(s,e)(obj.didSetMoviesSelected()));

      obj.listeners = listenerObjs;
      obj.hFig.DeleteFcn = @obj.lclDeleteFig;
      
      mainFigurePosition = obj.parent_.mainFigurePixelPosition() ;
      centerOnOtherFigureGivenPositionBang(obj.hFig, mainFigurePosition) ;
      lObj.syncMovieAndTrxFileExistence() ;
      obj.hFig.Visible = 'on';
      waitForFigureToSync(obj.hFig) ;
    end

    function delete(obj)
      % Destructor: tear down the figure and the Labeler listeners.
      delete(obj.hFig);
      for i=1:numel(obj.listeners)
        delete(obj.listeners{i});
      end
      obj.listeners = [];      
    end
    
    function lclDeleteFig(obj,~,~)
      % Delete the Labeler listeners when the MMC figure is closed.
      listenerObjs = obj.listeners;
      for i=1:numel(listenerObjs)
        listener = listenerObjs{i};
        if isvalid(listener)
          delete(listener);
        end
      end
    end
    
    function rowheights = getGridLayoutRowHeights_(obj)
      % Compute the RowHeight cell array for the top-level uigridlayout.
      lObj = obj.labeler;
      if lObj.nview == 1,
        rowheights = {'1x',0,0,40};
      else
        rowheights = {'1x',20,20*(lObj.nview+1),40};
      end
    end

    function selectionChangedTblMovies(obj,~,~)
      % Actuation: tblMain selection changed; write the new moviesSelected.
      % The selected column is pure view state (not in the model), so we
      % refresh the Change Path enablement directly --- no didSet*
      % notification will drive it for us.
      obj.labeler.moviesSelected = obj.getSelectedMovies_() ;
      obj.updateChangePathButtonEnablement_() ;
    end

    function selectionChangedTblMovieSet(obj,~,~)
      % Actuation: tblView selection changed.  No model state to update
      % here --- the per-view selection only matters to the Change Path
      % button --- so just refresh that button's enablement.
      obj.updateChangePathButtonEnablement_() ;
    end  % function

    function doubleClickFcnCallbackTblMovies(obj,~,evt)
      % Actuation: double-click on a movie row switches the Labeler to it.
      % This should have the same action as first selecting the row and them
      % clicking the "Switch to Movie" button.
      row = evt.InteractionInformation.DisplayRow;
      if isempty(row)
        return
      end
      obj.labeler.movieSet(row);
    end
    
    function setVisible(obj, tf)
      % Show or hide the MovieManager figure.
      wasVisible = strcmp(obj.hFig.Visible, 'on') ;
      obj.hFig.Visible = onIff(tf);
      if tf
        if ~wasVisible
          % Refresh on-disk file existence so any out-of-band changes
          % since the figure was last visible are reflected.
          obj.labeler.syncMovieAndTrxFileExistence() ;
        end
        figure(obj.hFig);
      end
      waitForFigureToSync(obj.hFig) ;
    end

    function tf = isValid(obj)
      % True if the MovieManager figure still exists.
      tf = isvalid(obj.hFig);
    end

    function idx = getSelectedMovies_(obj)
      % Get the indices of movies currently selected in obj.tblMain.

      selection = obj.tblMain.Selection ;
      % MATLAB sometimes hands back Selection in a degenerate empty
      % shape like 1x0 (e.g. clicks in empty space), so guard.
      if isempty(selection)
        idx = zeros(0,1) ;
      else
        idx = unique(selection(:,1),'stable') ;
      end
    end
    
    function cbkPushButton(obj,src,~)
      % Actuation: dispatch a button press to the matching action.
      lObj = obj.labeler;

      switch src.Tag
        case 'pbAdd'
          obj.addLabelerMovie(); % can throw
        case 'pbRm'
          obj.rmLabelerMovie();
        case 'pbChangePath'
          obj.changeLabelerPath_();
        case 'pbSwitch'
          if obj.labeler.gtIsGTMode
            obj.parent_.gtShowGTManager();
          else
            iMov = obj.getSelectedMovies_();
            if ~isempty(iMov)
              iMov = iMov(1);
              obj.labeler.movieSet(iMov)
            end
          end
        case 'pbNextUnlabeled'
          iMov = find(~lObj.movieFilesAllHaveLbls,1);
          if isempty(iMov)
            msgbox('All movies are labeled!');
          else
            lObj.movieSet(iMov);
          end
        otherwise
          assert(false);
      end
    end
  
    function mnuFileAddMoviesBatch(obj)
      % Actuation: prompt for a batch file and add the listed movies.
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
        lObj.movieSet(1);
      end
    end

    function bringWindowToFront(obj)
      % Make the MovieManager figure visible and bring it to the foreground.
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
      % Sync every aspect of the MMC GUI to the current Labeler state.
      obj.updatePointer() ;
      obj.updateMovieData_();
      obj.updateMostButtonEnablement_();
      obj.updateTableSelection_();
      obj.updateMovieSetDetails_();
      obj.updateMenuEnablement_();
      obj.updateChangePathButtonEnablement_() ;
    end

    function tf = isLabelerOutOfInitAndHasProject_(obj)
      % Whether MMC controls should be enabled, given Labeler state.
      lObj = obj.labeler ;
      tf = ~lObj.isinit && lObj.hasProject ;
    end

    function updateMostButtonEnablement_(obj)
      % Sync the bottom-row push buttons (text and Enable) to Labeler state.
      lObj = obj.labeler ;
      areControlsEnabled = obj.isLabelerOutOfInitAndHasProject_() ;
      if lObj.gtIsGTMode,
        set(obj.pbSwitch,'Text','GT Frames');
        obj.pbNextUnlabeled.Visible = 'off';
      else
        set(obj.pbSwitch,'Text','Switch to Movie');
        obj.pbNextUnlabeled.Visible = 'on';
      end
      obj.pbSwitch.Enable = onIff(areControlsEnabled) ;
      obj.pbNextUnlabeled.Enable = onIff(areControlsEnabled) ;
      obj.pbAdd.Enable = onIff(areControlsEnabled) ;
      obj.pbRm.Enable = onIff(areControlsEnabled) ;
    end

    function updateChangePathButtonEnablement_(obj)
      % Sync pbChangePath.Enable to whether a single, concrete path cell
      % is selected.  In multiview projects a click in the main table
      % only identifies a movieset (n paths), so the button stays
      % disabled until the user picks a specific view in the per-view
      % table.
      if obj.isLabelerOutOfInitAndHasProject_()
        [~, ~, ~, isPathCellSelected] = obj.determineSelectedPathCell_() ;
        doEnable = isPathCellSelected ;
      else
        doEnable = false ;
      end
      obj.pbChangePath.Enable = onIff(doEnable) ;
    end  % function

    function updateMenuEnablement_(obj)
      % Sync the File menu items' Enable state to the Labeler.
      areControlsEnabled = obj.isLabelerOutOfInitAndHasProject_() ;
      obj.menuFileAddMoviesFromTextFile.Enable = onIff(areControlsEnabled) ;
    end
    
    function updateTableSelection_(obj)
      % Sync tblMain.Selection to labeler.moviesSelected, preserving
      % each row's currently-selected column where possible.
      selectedMovies = obj.labeler.moviesSelected ;
      if isempty(selectedMovies) || isempty(obj.tblMain.Data)
        obj.tblMain.Selection = zeros(0,2) ;
      else
        currentSelection = obj.tblMain.Selection ;
        n = numel(selectedMovies) ;
        cols = ones(n,1) ;
        if size(currentSelection,2) >= 2
          [tf,loc] = ismember(selectedMovies(:), currentSelection(:,1)) ;
          cols(tf) = currentSelection(loc(tf), 2) ;
        end
        obj.tblMain.Selection = [selectedMovies(:), cols] ;
      end
    end

    function didSetMoviesSelected(obj)
      % Listener callaback for didSetMoviesSelected event in the Labeler.
      obj.updateTableSelection_() ;
      obj.updateMovieSetDetails_() ;
      obj.updateChangePathButtonEnablement_() ;
    end

    function updateMovieSetDetails_(obj)
      % Sync per-view detail pane (obj.tblView and obj.labelSet) to
      % labeler.moviesSelected.
      lObj = obj.labeler ;
      rows = lObj.moviesSelected ;
      isMultiView = lObj.nview > 1 ;
      if isMultiView && numel(rows) == 1 && obj.isLabelerOutOfInitAndHasProject_()
        obj.tblView.Data = lObj.movieFilesAllGTaware(rows,:)' ;
        obj.labelSet.Text = sprintf('Selected movieset %d', rows) ;
      else
        obj.tblView.Data = cell(0,1) ;
        obj.labelSet.Text = '' ;
      end
    end

    function updateMovieData_(obj)
      % Sync obj.tblMain contents and visibility to the current Labeler state.
      lObj = obj.labeler ;
      areControlsEnabled = obj.isLabelerOutOfInitAndHasProject_() ;

      obj.gl.RowHeight = obj.getGridLayoutRowHeights_() ;
      obj.tblMain.Enable = onIff(areControlsEnabled) ;
      isMultiView = (lObj.nview > 1) ;
      obj.tblView.Visible = onIff(isMultiView) ;
      obj.labelSet.Visible = onIff(isMultiView) ;
      obj.tblView.Enable = onIff(isMultiView&&areControlsEnabled) ;

      if areControlsEnabled
        movNames = lObj.movieFilesAllGTaware ;
        trxNames = lObj.trxFilesAllGTaware ;
        movsHaveLbls = lObj.movieFilesAllHaveLblsGTaware ;
      else
        % No project / mid-init; force empty.
        movNames = cell(0,1) ;
        trxNames = cell(0,1) ;
        movsHaveLbls = false(0,1) ;
      end

      % In multiview projects each row represents a movieset (n movies,
      % one per view); in single-view it's just one movie per row.
      movieColumnHeader = fif(isMultiView, 'Movieset', 'Movie') ;
      JTABLEPROPS_NOTRX = {'ColumnName',{movieColumnHeader 'Num Labels'},...
                           'ColumnWidth',{'1x',250}};
      JTABLEPROPS_TRX = {'ColumnName',{movieColumnHeader 'Trx' 'Num Labels'},...
                         'ColumnWidth',{'2x','1x',100}};
      
      if isequal(size(movNames,1),size(trxNames,1),numel(movsHaveLbls))
        movSetNames = movNames(:,1) ;
        trxSetNames = trxNames(:,1) ;
        tfTrx = any(cellfun(@(x)~isempty(x),trxNames(:))) ;
        if tfTrx
          tableData = [movSetNames trxSetNames num2cell(int64(movsHaveLbls))] ;
          args = JTABLEPROPS_TRX ;
        else
          tableData = [movSetNames num2cell(int64(movsHaveLbls))] ;
          args = JTABLEPROPS_NOTRX ;
        end
      else
        % Model is mid-mutation; render empty until next event arrives.
        tableData = cell(0,2) ;
        args = JTABLEPROPS_NOTRX ;
      end
      set(obj.tblMain, args{:}, 'Data', tableData) ;
      obj.updateMovieFileExistenceStyles_() ;
    end

    function updateMovieFileExistenceStyles_(obj)
      % Color movie/trx cells in tblMain light red if the underlying
      % file did not exist on disk at the last check; default (white)
      % otherwise.  If the existence-prop size does not match the
      % currently displayed rows (e.g. stale after a GT mode toggle),
      % skip styling for that column.
      removeStyle(obj.tblMain) ;
      [rowCount, colCount] = size(obj.tblMain.Data) ;
      if rowCount == 0
        return
      end
      lObj = obj.labeler ;
      redStyle = uistyle('BackgroundColor', [1.0, 0.85, 0.85]) ;

      movieExists = lObj.doesMovieFileExist ;
      if size(movieExists, 1) == rowCount
        missingRows = find(any(~movieExists, 2)) ;
        if ~isempty(missingRows)
          cells = [missingRows, ones(numel(missingRows), 1)] ;
          addStyle(obj.tblMain, redStyle, 'cell', cells) ;
        end
      end

      hasTrxColumn = (colCount == 3) ;
      if hasTrxColumn
        trxExists = lObj.doesTrxFileExist ;
        if size(trxExists, 1) == rowCount
          missingRows = find(any(~trxExists, 2)) ;
          if ~isempty(missingRows)
            cells = [missingRows, 2*ones(numel(missingRows), 1)] ;
            addStyle(obj.tblMain, redStyle, 'cell', cells) ;
          end
        end
      end
    end  % function

    function addLabelerMovie(obj)
      % Actuation: prompt the user for a movie (or movie set) and add it.
      lObj = obj.labeler;
      nmovieOrig = lObj.nmoviesGTaware;
      if lObj.nview==1
        % Project is single-view
        [tfsucc,movfile,trxfile] = promptGetMovTrxFiles(true,lObj.projectHasTrx);
        if ~tfsucc
          return
        end
        try
          lObj.movieAdd(movfile,trxfile);
        catch ME,
          uiwait(errordlg(getReport(ME,'basic','hyperlinks','off'),'Error adding movies'));
          return
        end
      else
        % Project is multi-view
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
          return
        end
        try
          lObj.movieSetAdd(movfiles);
        catch ME,
          uiwait(errordlg(getReport(ME,'basic','hyperlinks','off'),'Error adding movies'));
          return
        end
      end
      if nmovieOrig==0 && lObj.nmoviesGTaware>0
        lObj.movieSet(1);
      end
    end
    
    function changeLabelerPath_(obj)
      % Actuation: change the path of the currently selected movie or trx
      % cell, with optional prefix-aware fanout to other matching paths.
      [iMov, iView, isMovie, isValid, errMsg] = obj.determineSelectedPathCell_() ;
      if ~isValid
        uiwait(errordlg(errMsg, 'No path cell selected')) ;
        return
      end
      isGT = obj.labeler.gtIsGTMode ;
      try
        obj.parent_.changeMovieOrTrxFilePathGUI(iMov, iView, isGT, isMovie) ;
      catch ME
        uiwait(errordlg(getReport(ME, 'basic', 'hyperlinks', 'off'), 'Error changing path')) ;
      end
    end  % function

    function [iMov, iView, isMovie, isValid, errMsg] = determineSelectedPathCell_(obj)
      % Determine which movie/trx cell the user has selected.  Returns
      % isValid=false (and a user-readable errMsg) if no suitable cell is
      % selected.
      iMov = 0 ;
      iView = 0 ;
      isMovie = false ;
      isValid = false ;
      errMsg = '' ;

      lObj = obj.labeler ;
      if lObj.nview==1
        % For single-view, the only (visible) table is the main one
        sel = obj.tblMain.Selection ;
        if isempty(sel) || size(sel, 1) ~= 1
          errMsg = 'Select a single movie or trx cell first.' ;
          return
        end
        iMov = sel(1, 1) ;
        colIndex = sel(1, 2) ;
        colNames = obj.tblMain.ColumnName ;
        if colIndex < 1 || colIndex > numel(colNames)
          errMsg = 'Select a movie or trx cell.' ;
          return
        end
        colName = colNames{colIndex} ;
        switch colName
          case 'Movie'
            isMovie = true ;
          case 'Trx'
            isMovie = false ;
          case 'Movieset'
            % Multiview: a movieset row covers nview cells.  Make the
            % user pick a specific view in the per-view table below.
            errMsg = ['For multiview projects, select a per-view cell ' ...
                      'in the lower table to change a path.'] ;
            return
          otherwise
            errMsg = 'Select a movie or trx cell (not the labels column).' ;
            return
        end
        iView = 1 ;
      else
        % For multiview, the view table determines the selected movie
        sel = obj.tblView.Selection ;
        if isempty(sel) || size(sel, 1) ~= 1
          errMsg = 'Select a single per-view cell first.' ;
          return
        end
        iView = sel(1, 1) ;
        mvSel = obj.labeler.moviesSelected ;
        if numel(mvSel) ~= 1
          errMsg = 'Select exactly one movieset first.' ;
          return
        end
        iMov = mvSel(1) ;
        isMovie = true ;  % per-view table only shows movies
      end
      isValid = true ;
    end  % function

    function rmLabelerMovie(obj)
      % Actuation: remove the currently selected movies from the Labeler.
      selRow = sort(obj.getSelectedMovies_()) ;
      n = numel(selRow);
      labelerController = obj.parent_ ;
      for i = n:-1:1
        row = selRow(i);
        try
          tfSucc = labelerController.movieRmGUI(row);
        catch ME,
          uiwait(errordlg(getReport(ME,'basic','hyperlinks','off'),'Error removing movie'));
          break;
        end
        if ~tfSucc
          % user stopped/canceled
          break;
        end
      end
    end  % function    
  end  % methods (Hidden)  
end  % classdef
