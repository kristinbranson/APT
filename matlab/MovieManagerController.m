classdef MovieManagerController < handle
  properties (SetAccess=private)
    hFig % scalar handle to MovieManager fig
    
    labeler % scalar labeler Obj
    listeners % cell array of listener objs
    
    hTG % tab group
    hTabs % [2] uitabs
    mmTbls % [2] MovieManagerTables
    tblMovies
    tblMovieSet
    tabHandles % [2] "handles" struct array
    hPBs % [2] cell array, convenience handles. hPBs{1/2} contains handle array of reg/gt pb handles
  end
  properties (Constant)
    JTABLEPROPS_NOTRX = {'ColumnName',{'Movie' 'Num Labels'},...
                         'ColumnWidth',{'1x',250}};
    JTABLEPROPS_TRX = {'ColumnName',{'Movie' 'Trx' 'Num Labels'},...
                       'ColumnPreferredWidth',{'2x','1x',100}};
  end
  properties (Dependent)
    gtTabSelected % true if GT tab is current tab selection
    selectedTabMatchesLabelerGTMode % if true, the current tab selection is consistent with .labeler.gtIsGTMode
    mmTblCurr % element of .mmTbls for given .gtSelected
  end
  
  events
    tableClicked % fires when any movietable is clicked
  end
  
  methods
    function v = get.gtTabSelected(obj)
      v = obj.hTG.SelectedTab==obj.hTabs(2);
    end
    function v = get.selectedTabMatchesLabelerGTMode(obj)
      v = obj.gtTabSelected==obj.labeler.gtIsGTMode;
    end
    function v = get.mmTblCurr(obj)
      v = obj.getMMTblGT(obj.gtTabSelected);
    end
    function v = getMMTblGT(obj,gt)
      if gt
        v = obj.mmTbls(2);
      else
        v = obj.mmTbls(1);
      end
    end
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

      if lObj.nview == 1,
        gl_args = {[2 1],'RowHeight',{'1x',40}};
      else
        gl_args = {[3 1],'RowHeight',{'1x',20*(lObj.nview+1),40}};
      end
      handles.gl = uigridlayout(obj.hFig,gl_args{:},'tag','gl');

      handles.tblMovies = uitable(handles.gl,...
        'ColumnName',{'Movie','Has Lbls'},...
        'ColumnWidth',{'1x',70},...
        'tag','tblMovies',...
        'ButtonDownFcn',@(src,evt) obj.buttonDownFcnTblMovies(src,evt),...
        'CellSelectionCallback',@(src,evt) obj.cellSelectionCallbackTblMovies(src,evt));
      obj.tblMovies = handles.tblMovies;
      handles.gl_buttons = uigridlayout(handles.gl,[1,4],'Padding',[0,0,0,0],'tag','gl_buttons');

      if lObj.gtIsGTMode,
        handles.pbGTFrames = uibutton(handles.gl_buttons,'Text','GT Frames','tag','pbGTFrames',...
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
      handles.pbRm = uibutton(handles.gl_buttons,'Text','Remove Movie','tag','pbRemove',...
        'ButtonPushedFcn',@(src,evt) cbkPushButton(obj,src,evt));

      if lObj.nview > 1,
        handles.tblMovieSet = uitable(handles.gl,...
          'ColumnName',{},'tag','tblMovieSet');
        obj.tblMovieSet = handles.tblMovieSet;
      else
        obj.tblMovieSet = [];
      end

      handles.menu_file = uimenu('Tag','menu_file','Text','File','Parent',obj.hFig);
      handles.menu_file_add_movies_from_text_file = uimenu('Tag','menu_file_add_movies_from_text_file',...
        'Text','Add movies from text file','Parent',handles.menu_file);
      handles.menu_help = uimenu('Tag','menu_help','Text','Help','Parent',obj.hFig);

      handles.menu_file_add_movies_from_text_file.MenuSelectedFcn = ...
          @(s,e)obj.mnuFileAddMoviesBatch();

      guidata(obj.hFig,handles);

      set(obj.hFig,'MenuBar','None');
      obj.hFig.Visible = 'off';
      obj.hFig.DeleteFcn = @(s,e)delete(mmc);
      
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

      % TODO figure out 
      % lObjs{end+1,1} = addlistener(handles.tblMovies,'tableClicked',...
      %   @(s,e)obj.notify('tableClicked'));

      obj.listeners = lObjs;
            
      centerfig(obj.hFig,obj.labeler.gdata.mainFigure_);
    end
    
    function buttonDownFcnTblMovies(obj,src,evt)
    end

    function cellSelectionCallbackTblMovies(obj,src,evt)
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
      
      delete(obj.mmTbls);
      obj.mmTbls = [];
    end
    
    % function tabSetup(obj)
    % 
    %   tblReg = MovieManagerTable.create(obj.labeler.nview,tblOrig.Parent,...
    %     tblOrig.Position,@(iMov)obj.tblCbkMovieSelected(iMov));
    %   obj.listeners{end+1,1} = addlistener(tblReg,'tableClicked',...
    %     @(s,e)obj.notify('tableClicked'));
    % 
    %   obj.mmTbls = [tblReg tblGT];
    %   obj.tabHandles = gdata;
    %   obj.hPBs = hpbs;
    % 
    %   obj.selectTab(1);
    % end
        
  end
  
  methods
    
    function setVisible(obj,tf)
      onoff = onIff(tf);
      obj.hFig.Visible = onoff;
      if tf
        figure(obj.hFig);
      end
    end
        
    function mIdx = getSelectedMovies(obj)
      iMovs = obj.mmTblCurr.getSelectedMovies();
      mIdx = MovieIndex(iMovs,obj.gtTabSelected);
    end
    
    function cbkTabGrpSelChanged(obj,src,~)
      % When ~obj.labeler.gtIsGTMode, only tab1 can be selected
      % When obj.labeler.gtIsGTMode, both tabs can be selected
      
      iSelTab = find(src.SelectedTab==obj.hTabs);
      switch iSelTab
        case 1
          % none
        case 2
          if ~obj.labeler.gtIsGTMode
            src.SelectedTab = obj.hTabs(1);
            error('MovieManager:gt','GT mode is not enabled.');
          end
        otherwise
          assert(false);
      end
      obj.updateMenusEnable();
    end
    
    function tblCbkMovieSelected(obj,iMov)
      assert(isscalar(iMov) && iMov>0);
      % iMov is gt-aware movie index (unsigned)
      lObj = obj.labeler;
      if obj.selectedTabMatchesLabelerGTMode
        lObj.movieSetGUI(iMov);
      else
        if lObj.gtIsGTMode
          warnstr = 'Labeler is in GT mode; select ''GT'' Tab in Movie Manager if you wish to browse movies via the table.';
        else
          warnstr = 'Labeler is not in GT mode; select ''Regular'' Tab in Movie Manager if you wish to browse movies via the table.';
        end
        warningNoTrace('MovieManagerController:nav',warnstr);
      end
    end
    
    function cbkPushButton(obj,src,~)
      iTab = find(src.Parent==obj.hTabs);
      lObj = obj.labeler;
      tfGT = lObj.gtIsGTMode;
      assert(iTab==double(tfGT)+1); % "wrong tab" buttons are disabled
      
      switch src.Tag
        case 'pbAdd'
          obj.addLabelerMovie(); % can throw
        case 'pbRm'
          obj.rmLabelerMovie();
        case 'pbSwitch' 
          iMov = obj.mmTblCurr.getSelectedMovies();
          if ~isempty(iMov)
            iMov = iMov(1);
            obj.tblCbkMovieSelected(iMov);
          end
        case 'pbNextUnlabeled'
          assert(~tfGT);
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
      obj.hlpLblerLstnCbkUpdateTable(false);
    end
    
    function lblerLstnCbkUpdateTableGT(obj,~,~)
      obj.hlpLblerLstnCbkUpdateTable(true);      
    end
    
    function lblerLstnCbkProjLoaded(obj,~,~)
      obj.hlpLblerLstnCbkUpdateTable(false);
      obj.hlpLblerLstnCbkUpdateTable(true);  % Me no like.  -- ALT, 2025-02-11
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
      
      if tfGT
        obj.showGTTab();
      end
      obj.updatePushButtonsEnable(tfGT);
      obj.updateMMTblRowSelection();
      iTab = double(tfGT)+1;
      obj.selectTab(iTab);      
      if ~tfGT
        obj.hideGTTab();
      end
      
      obj.updateMenusEnable();
    end
    
    function mnuFileAddMoviesBatch(obj)
      assert(obj.selectedTabMatchesLabelerGTMode);
      lastTxtFile = RC.getprop('lastMovieBatchFile');
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
      
      lObj = obj.labeler;
      nmovieOrig = lObj.nmoviesGTaware;
      fname = fullfile(pname,fname);
      lObj.movieAddBatchFile(fname);
      RC.saveprop('lastMovieBatchFile',fname);
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
      labeler = obj.labeler ;
      is_busy = labeler.isStatusBusy ;
      pointer = fif(is_busy, 'watch', 'arrow') ;
      set(obj.hFig, 'Pointer', pointer) ;
    end  % function    
  end  % methods
  
  methods (Hidden)
    
    function showGTTab(obj)
      obj.hTabs(2).Parent = obj.hTG;
    end
    function hideGTTab(obj)
      obj.hTabs(2).Parent = [];
    end
    
    function selectTab(obj,iTab)
      assert(any(iTab==1:2));
      obj.hTG.SelectedTab = obj.hTabs(iTab);
    end
    
    function updatePushButtonsEnable(obj,tfGT)
      if tfGT
        set(obj.hPBs{1},'Enable','off');
        set(obj.hPBs{2},'Enable','on');
      else
        set(obj.hPBs{1},'Enable','on');
        set(obj.hPBs{2},'Enable','off');
      end
    end
    
    function updateMenusEnable(obj)
      onoff = onIff(obj.selectedTabMatchesLabelerGTMode);
      gdata = guidata(obj.hFig);
      gdata.menu_file_add_movies_from_text_file.Enable = onoff;
    end
    
    function setSelectedMovie(obj,iMov)
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

      szassert(trxNames,size(movNames));
      assert(size(movNames,1)==numel(movsHaveLbls));


      nSets = size(movNames,1);
      assert(size(movNames,2)==obj.labeler.nview);
      assert(nSets==numel(movsHaveLbls));
      
      movNames = movNames';
      movsHaveLbls = repmat(movsHaveLbls(:),1,obj.nmovsPerSet);
      movsHaveLbls = movsHaveLbls';
      dat = [num2cell(iSet(:)) movNames(:) num2cell(movsHaveLbls(:))];
      
      tt = treeTable(obj.hParent,obj.HEADERS,dat,...
        'ColumnTypes',obj.COLTYPES,...
        'ColumnEditable',obj.COLEDIT,...
        'Groupable',true,...
        'IconFilenames',...
            {'' fullfile(matlabroot,'/toolbox/matlab/icons/file_open.png') fullfile(matlabroot,'/toolbox/matlab/icons/foldericon.gif')});
      cwMap = obj.COLWIDTHS;
      keys = cwMap.keys;
      for k=keys(:)',k=k{1}; %#ok<FXSET>
        tblCol = tt.getColumn(k);
        tblCol.setPreferredWidth(cwMap(k));
      end
      
      tt.MouseClickedCallback = @(s,e)obj.cbkClickedDefault(s,e);
      tt.setDoubleClickEnabled(false);
      if ~isempty(obj.tbl)
        delete(obj.tbl);
      end
      obj.tbl = tt;

      
      tfTrx = any(cellfun(@(x)~isempty(x),trxNames));
      if tfTrx
        assert(size(trxNames,2)==1,'Expect single column.');
        dat = [movNames trxNames num2cell(int64(movsHaveLbls))];
        args = MovieManagerController.JTABLEPROPS_TRX;
      else
        dat = [movNames num2cell(int64(movsHaveLbls))];
        args = MovieManagerController.JTABLEPROPS_NOTRX;
      end





      set(obj.tblMovies,args{:},'Data',dat);

    end
    
    function hlpLblerLstnCbkUpdateTable(obj,tfGT)
      lObj = obj.labeler;
      if lObj.isinit
        return
      end
      if ~exist('tfGT', 'var') || isempty(tfGT) ,
        tfGT = lObj.gtIsGTMode ;
      end

      assert(islogical(tfGT));     

      if ~lObj.hasProject
        return
        % error('MovieManagerController:proj',...
        %   'Please open/create a project first.');
      end
      
      PROPS = Labeler.gtGetSharedPropsStc(tfGT);
      movs = lObj.(PROPS.MFA);
      trxs = lObj.(PROPS.TFA);
      movsHaveLbls = lObj.(PROPS.MFAHL);
      if ~isequal(size(movs,1),size(trxs,1),numel(movsHaveLbls))
        % intermediate state, take no action
        return
      end
      
      obj.updateMovieData(movs,trxs,movsHaveLbls);
      if tfGT==lObj.gtIsGTMode
        % Not conditional is necessary, could just always update
        obj.updateMMTblRowSelection();
      end
    end
    
    function addLabelerMovie(obj)
      lObj = obj.labeler;
      nmovieOrig = lObj.nmoviesGTaware;
      if lObj.nview==1
        [tfsucc,movfile,trxfile] = promptGetMovTrxFiles(true,lObj.projectHasTrx);
        if ~tfsucc
          return;
        end
        lObj.movieAdd(movfile,trxfile);
      else
        assert(lObj.nTargets==1,'Adding trx files currently unsupported.');
        lastmov = RC.getprop('lbl_lastmovie');
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
        lObj.movieSetAdd(movfiles);
      end
      if nmovieOrig==0 && lObj.nmoviesGTaware>0
        lObj.movieSetGUI(1,'isFirstMovie',true);
      end
    end
    
    function rmLabelerMovie(obj)
      selRow = obj.mmTblCurr.getSelectedMovies();
      selRow = sort(selRow);
      n = numel(selRow);
      lObj = obj.labeler;
      for i = n:-1:1
        row = selRow(i);
        tfSucc = lObj.movieRmGUI(row);
        if ~tfSucc
          % user stopped/canceled
          break;
        end
      end
    end
    
  end  % methods (Hidden)
  
end  % classdef