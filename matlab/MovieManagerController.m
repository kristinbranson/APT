classdef MovieManagerController < handle
  properties (SetAccess=private)
    parent_ % scalar LabelerController that created this object
    hFig % scalar handle to MovieManager fig

    labeler % scalar labeler Obj
    listeners % cell array of listener objs

    tblMovies
    tblMovieSet
    tabHandles % [2] "handles" struct array

    % UI handles previously kept in guidata(hFig).
    gl  % top-level uigridlayout
    labelSet  % uilabel showing which movieset is selected
    glButtons  % uigridlayout for the bottom button row
    pbSwitch  % uibutton: "Switch to Movie" or "GT Frames"
    pbNextUnlabeled  % uibutton: "Next Unlabeled"
    pbAdd  % uibutton: "Add Movie"
    pbRm  % uibutton: "Remove Movie"
    menuFile  % uimenu: "File"
    menuFileAddMoviesFromTextFile  % uimenu under File
  end

  properties (Constant)
    JTABLEPROPS_NOTRX = {'ColumnName',{'Movie' 'Num Labels'},...
                         'ColumnWidth',{'1x',250}};
    JTABLEPROPS_TRX = {'ColumnName',{'Movie' 'Trx' 'Num Labels'},...
                       'ColumnWidth',{'2x','1x',100}};
  end
    
  methods
    
    % MovieManagerController messages between Labeler and Tables
    % 1. Labeler/clients can fetch current selection in Table
    % 2. Labeler prop changes fire MMC listeners to update Table content
    % 3. MMC Tables can set current movie in Labeler based on user action
    % 4. MMC buttons can add/rm labeler movies

    function obj = MovieManagerController(labelerController, labeler)
      assert(isa(labelerController, 'LabelerController'));
      assert(isa(labeler, 'Labeler'));
      obj.parent_ = labelerController;
      lObj = labeler;
      %obj.hFig = MovieManager(obj);
      obj.labeler = lObj;
      
      obj.hFig = uifigure('Units','pixels','Position',[951,1400,733,436],...
        'Name','Manage Movies');
      %obj.hFig.CloseRequestFcn = @(hObject,eventdata) obj.CloseRequestFcn(hObject,eventdata);

      obj.gl = uigridlayout(obj.hFig,[4,1],'RowHeight',obj.getGridLayoutRowHeights(),'tag','gl');

      obj.tblMovies = uitable(obj.gl,...
        'ColumnName',{'Movie','Has Lbls'},...
        'ColumnWidth',{'1x',70},...
        'tag','tblMovies',...
        'CellSelectionCallback',@(src,evt) obj.cellSelectionCallbackTblMovies(src,evt),...
        'DoubleClickedFcn',@(src,evt) obj.doubleClickFcnCallbackTblMovies(src,evt));
      obj.labelSet = ...
        uilabel('Parent',obj.gl,...
                'Text','<no movieset selected>',...
                'Visible',onIff(lObj.nview > 1),...
                'HorizontalAlignment','center');

      rownames = arrayfun(@(x) sprintf('View %d',x), 1:lObj.nview,'Uni',0);
      obj.tblMovieSet = uitable(obj.gl,...
        'ColumnName',{},'tag','tblMovieSet',...
        'RowName',rownames,'Visible',onIff(lObj.nview > 1));

      obj.glButtons = uigridlayout(obj.gl,[1,4],'Padding',[0,0,0,0],'tag','gl_buttons');

      if lObj.gtIsGTMode,
        obj.pbSwitch = uibutton(obj.glButtons,'Text','GT Frames','tag','pbGTFrames',...
          'ButtonPushedFcn',@(src,evt) cbkPushButton(obj,src,evt));
      else
        obj.pbSwitch = uibutton(obj.glButtons,'Text','Switch to Movie','tag','pbSwitch',...
          'ButtonPushedFcn',@(src,evt) cbkPushButton(obj,src,evt));
      end
      obj.pbNextUnlabeled = uibutton(obj.glButtons,'Text','Next Unlabeled','tag','pbNextUnlabeled',...
        'ButtonPushedFcn',@(src,evt) cbkPushButton(obj,src,evt));
      if lObj.gtIsGTMode,
        obj.pbNextUnlabeled.Visible = 'off';
      end

      obj.pbAdd = uibutton(obj.glButtons,'Text','Add Movie','tag','pbAdd',...
        'ButtonPushedFcn',@(src,evt) cbkPushButton(obj,src,evt));
      obj.pbRm = uibutton(obj.glButtons,'Text','Remove Movie','tag','pbRm',...
        'ButtonPushedFcn',@(src,evt) cbkPushButton(obj,src,evt));


      obj.menuFile = uimenu('Tag','menu_file','Text','File','Parent',obj.hFig);
      obj.menuFileAddMoviesFromTextFile = uimenu('Tag','menu_file_add_movies_from_text_file',...
        'Text','Add movies from text file','Parent',obj.menuFile);

      obj.menuFileAddMoviesFromTextFile.MenuSelectedFcn = ...
          @(s,e)obj.mnuFileAddMoviesBatch();

      set(obj.hFig,'MenuBar','None');
      obj.update();
      obj.hFig.Visible = 'on';
      
      lObjs = cell(0,1);
      lObjs{end+1,1} = addlistener(lObj,'didSetMovieFilesAll',@(s,e)(obj.update()));
      lObjs{end+1,1} = addlistener(lObj,'didSetMovieFilesAllHaveLbls',@(s,e)(obj.update()));
      lObjs{end+1,1} = addlistener(lObj,'didSetTrxFilesAll',@(s,e)(obj.update()));
      lObjs{end+1,1} = addlistener(lObj,'didSetMovieFilesAllGT',@(s,e)(obj.update()));
      lObjs{end+1,1} = addlistener(lObj,'didSetMovieFilesAllGTHaveLbls',@(s,e)(obj.update()));
      lObjs{end+1,1} = addlistener(lObj,'didSetTrxFilesAllGT',@(s,e)(obj.update()));
      lObjs{end+1,1} = addlistener(lObj,'didLoadProject',@(s,e)(obj.update()));
      lObjs{end+1,1} = addlistener(lObj,'gtIsGTModeChanged',@(s,e)(obj.update()));

      obj.listeners = lObjs;
      obj.hFig.DeleteFcn = @obj.lclDeleteFig;
      
      mainFigurePosition = obj.parent_.mainFigurePixelPosition() ;
      centerOnOtherFigureGivenPositionBang(obj.hFig, mainFigurePosition) ;
      waitForFigureToSync(obj.hFig) ;
    end

    function lclDeleteFig(obj,~,~)
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

    function cellSelectionCallbackTblMovies(obj,~,evt)
      rows = evt.Indices(:,1);
      if obj.labeler.nview > 1,
        if numel(rows) ~= 1,
          obj.tblMovieSet.Data = cell(0,1);
          obj.labelSet.Text = '';
          obj.labeler.moviesSelected = obj.getSelectedMovies() ;
          return;
        end
        obj.tblMovieSet.Data = obj.labeler.movieFilesAllGTaware(rows,:)';
        obj.labelSet.Text = sprintf('Selected movieset %d',rows);
      end
      obj.labeler.moviesSelected = obj.getSelectedMovies() ;
    end

    function doubleClickFcnCallbackTblMovies(obj,~,evt)
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
    
    function setVisible(obj, tf)
      obj.hFig.Visible = onIff(tf);
      if tf
        figure(obj.hFig);
      end
      waitForFigureToSync(obj.hFig) ;
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
      obj.labeler.movieSet(iMov);
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
            lObj.movieSet(iMov);
          end
        case 'pbGTFrames'
          obj.parent_.gtShowGTManager();
        otherwise
          assert(false);
      end
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
        lObj.movieSet(1);
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
      obj.updateMovieData();
      obj.updatePushButtonsEnable();
      obj.updateMMTblRowSelection();
      obj.updateMenusEnable();
    end
    
    function updatePushButtonsEnable(obj)
      lObj = obj.labeler;
      if lObj.gtIsGTMode,
        set(obj.pbSwitch,'Text','GT Frames','tag','pbGTFrames');
        obj.pbNextUnlabeled.Visible = 'off';
      else
        set(obj.pbSwitch,'Text','Switch to Movie','tag','pbSwitch');
        obj.pbNextUnlabeled.Visible = 'on';
      end
    end

    function updateMenusEnable(obj)
      obj.menuFileAddMoviesFromTextFile.Enable = 'on';
    end
    
    function updateMMTblRowSelection(obj)
      % Sync tblMovies.Selection to labeler.moviesSelected.
      selectedMovies = obj.labeler.moviesSelected ;
      if isempty(selectedMovies) || isempty(obj.tblMovies.Data)
        obj.tblMovies.Selection = zeros(0,2) ;
      else
        n = numel(selectedMovies) ;
        obj.tblMovies.Selection = [selectedMovies(:), ones(n,1)] ;
      end
    end

    function updateMovieData(obj)
      lObj = obj.labeler;

      obj.gl.RowHeight = obj.getGridLayoutRowHeights();
      obj.tblMovieSet.Visible = onIff(lObj.nview > 1);
      obj.labelSet.Visible = onIff(lObj.nview > 1);

      movNames = lObj.movieFilesAllGTaware;
      trxNames = lObj.trxFilesAllGTaware;
      movsHaveLbls = lObj.movieFilesAllHaveLblsGTaware;

      if isequal(size(movNames,1),size(trxNames,1),numel(movsHaveLbls))
        movSetNames = movNames(:,1);
        trxSetNames = trxNames(:,1);
        tfTrx = any(cellfun(@(x)~isempty(x),trxNames(:)));
        if tfTrx
          dat = [movSetNames trxSetNames num2cell(int64(movsHaveLbls))];
          args = MovieManagerController.JTABLEPROPS_TRX;
        else
          dat = [movSetNames num2cell(int64(movsHaveLbls))];
          args = MovieManagerController.JTABLEPROPS_NOTRX;
        end
      else
        % Model is mid-mutation; render empty until next event arrives.
        dat = cell(0,2);
        args = MovieManagerController.JTABLEPROPS_NOTRX;
      end
      set(obj.tblMovies,args{:},'Data',dat);
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
        lObj.movieSet(1,'isFirstMovie',true);
      end
    end
    
    function rmLabelerMovie(obj)
      selRow = obj.getSelectedMovies();
      selRow = sort(selRow);
      n = numel(selRow);
      % lObj = obj.labeler;
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
