classdef MovieManagerController < handle
  properties (SetAccess=private)
    hFig % scalar handle to MovieManager fig
    
    labeler % scalar labeler Obj
    listeners % cell array of listener objs
    
    tblMovies
    tblMovieSet
    tabHandles % [2] "handles" struct array
  end
  properties (Constant)
    JTABLEPROPS_NOTRX = {'ColumnName',{'Movie' 'Num Labels'},...
                         'ColumnWidth',{'1x',250}};
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
        'Name','Manage Movies');
      handles.figure1 = obj.hFig;
      obj.hFig.CloseRequestFcn = @(hObject,eventdata) obj.CloseRequestFcn(hObject,eventdata);

      handles.gl = uigridlayout(obj.hFig,[4,1],'RowHeight',obj.getGridLayoutRowHeights(),'tag','gl');

      handles.tblMovies = uitable(handles.gl,...
        'ColumnName',{'Movie','Has Lbls'},...
        'ColumnWidth',{'1x',70},...
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

      handles.gl_buttons = uigridlayout(handles.gl,[1,4],'Padding',[0,0,0,0],'tag','gl_buttons');

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
      handles.menu_help = uimenu('Tag','menu_help','Text','Help','Parent',obj.hFig);

      handles.menu_file_add_movies_from_text_file.MenuSelectedFcn = ...
          @(s,e)obj.mnuFileAddMoviesBatch();

      guidata(obj.hFig,handles);

      set(obj.hFig,'MenuBar','None');
      obj.hFig.Visible = 'off';
      
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

      %lObjs{end+1,1} = addlistener(lObj,'didLoadProject',@(s,e)obj.lblerLstnCbkProjLoaded(s,e));
      lObjs{end+1,1} = addlistener(lObj,'newMovie',@(s,e)obj.lblerLstnCbkNewMovie(s,e));
      lObjs{end+1,1} = addlistener(lObj,'gtIsGTModeChanged',@(s,e)obj.lblerLstnCbkGTMode(s,e));

      obj.listeners = lObjs;
      obj.hFig.DeleteFcn = @obj.lclDeleteFig;
      
      centerfig(obj.hFig,obj.labeler.gdata.mainFigure_);
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
        obj.tblMovieSet.Data = obj.labeler.movieFilesAllGTaware(rows,:)';
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

    function CloseRequestFcn(obj,src,evt)
      src.Visible = 'off';
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
      if isempty(iMov),
        obj.tblMovies.Selection = zeros(0,2);
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
      set(obj.tblMovies,args{:},'Data',dat);

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
    
  end  % methods (Hidden)
  
end  % classdef