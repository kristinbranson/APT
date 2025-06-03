classdef MovieManagerController < handle
  properties (SetAccess=private)
    hFig % scalar handle to MovieManager fig
    
    labeler % scalar labeler Obj
    listeners % cell array of listener objs
    
    hTG % tab group
    hTabs % [2] uitabs
    mmTbls % [2] MovieManagerTables
    tabHandles % [2] "handles" struct array
    hPBs % [2] cell array, convenience handles. hPBs{1/2} contains handle array of reg/gt pb handles
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
      obj.hFig = MovieManager(obj);
      %obj.gdata = guidata(obj.hFig);
      obj.labeler = lObj;
      
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
      
      obj.tabSetup();
      
      gdata = guidata(obj.hFig);
      gdata.menu_file_add_movies_from_text_file.Callback = ...
          @(s,e)obj.mnuFileAddMoviesBatch();
      
      centerfig(obj.hFig,obj.labeler.gdata.mainFigure_);
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
    
    function tabSetup(obj)
      obj.hTG = uitabgroup(obj.hFig,...
        'Position',[0 0 1 1],'Units','normalized',...
        'SelectionChangedFcn',@(s,e)obj.cbkTabGrpSelChanged(s,e));
      hT1 = uitab(obj.hTG,'Title','Movie List');
      hT2 = uitab(obj.hTG,'Title','GT Movie List');
      obj.hTabs = [hT1 hT2];
      
      gdata = struct();
      
      % Take everything in MovieManager and move it onto reg tab
      HANDLES = {'uipanel1' 'pbSwitch' 'pbNextUnlabeled' 'pbAdd' 'pbRm'};
      mmgd = guidata(obj.hFig);
      hpbs = cell(1,2);
      for tag=HANDLES,tag=tag{1}; %#ok<FXSET>
        h = mmgd.(tag);
        h.Tag = tag;
        h.Parent = hT1;
        gdata(1).(tag) = h;
        if isequal(tag(1:2),'pb')
          h.Callback = @(s,e)obj.cbkPushButton(s,e);
          hpbs{1}(end+1,1) = h;
        end
      end
      tblOrig = mmgd.tblMovies;
      tblOrig.Visible = 'off';
      tblReg = MovieManagerTable.create(obj.labeler.nview,tblOrig.Parent,...
        tblOrig.Position,@(iMov)obj.tblCbkMovieSelected(iMov));
      obj.listeners{end+1,1} = addlistener(tblReg,'tableClicked',...
        @(s,e)obj.notify('tableClicked'));
      
      % Copy stuff onto GT tab
      HANDLES = {'uipanel1' 'pbSwitch' 'pbNextUnlabeled' 'pbAdd' 'pbRm'};
      for tag=HANDLES,tag=tag{1}; %#ok<FXSET>
        h = copyobj(mmgd.(tag),hT2);
        h.Tag = tag;
        gdata(2).(tag) = h;
        if isequal(tag(1:2),'pb')
          h.Callback = @(s,e)obj.cbkPushButton(s,e);
          hpbs{2}(end+1,1) = h;
        end
      end
      % Special case, pbNextUnlabeled->GT Frames
      gdata(2).pbNextUnlabeled.String = 'GT Frames';
      gdata(2).pbNextUnlabeled.Tag = 'pbGTFrames';
      gdata(2).pbGTFrames = gdata(2).pbNextUnlabeled;
      gdata(2).pbNextUnlabeled = [];
      tblGT = MovieManagerTable.create(obj.labeler.nview,gdata(2).uipanel1,...
        tblOrig.Position,@(iMov)obj.tblCbkMovieSelected(iMov));
      obj.listeners{end+1,1} = addlistener(tblGT,'tableClicked',...
        @(s,e)obj.notify('tableClicked'));
      
      obj.mmTbls = [tblReg tblGT];
      obj.tabHandles = gdata;
      obj.hPBs = hpbs;
      
      obj.selectTab(1);
    end
        
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
    
    function updateMMTblRowSelection(obj)
      % Update one of the MM tables per lObj.currMovie, lObj.gtIsGTMode
      
      lObj = obj.labeler;
      tfGT = lObj.gtIsGTMode;
      tbl = obj.getMMTblGT(tfGT);
      iMov = lObj.currMovie;
      if ~isempty(iMov)
        % - first clause: this can occur during projload
        tbl.updateSelectedMovie(iMov);
      end
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
      
      iTbl = double(tfGT)+1;
      tbl = obj.mmTbls(iTbl);
      tbl.updateMovieData(movs,trxs,movsHaveLbls);
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