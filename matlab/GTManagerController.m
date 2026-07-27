classdef GTManagerController < handle
  % Controller for the "Groundtruth Navigator" window, which lists the
  % groundtruth to-label suggestions per movie/frame/target and lets the user
  % navigate to them and compute GT accuracy.
  %
  % This replaces the bare uifigure function GTManager.m: state and widget
  % handles move from guidata into instance properties, and the Labeler
  % listeners that keep the tables in sync are owned by the controller (torn
  % down in delete()).  The figure is still built with modern uifigure
  % components.
  %
  % GTManager<->Labeler messaging:
  %  - the controller sets current movie/frame/target in the Labeler based on
  %    table interaction;
  %  - Labeler event listeners refresh the table content, selection, and
  %    enabled state.

  properties (Transient)
    hFig  % the window figure
    parent_  % a LabelerController (provides mainFigure_)
    labeler_  % a Labeler
    gl_  % top-level uigridlayout
    gl_buttons_  % button-row uigridlayout
    tblGTMovie_  % movie-list uitable
    tblFrame_  % frame/target uitable
    pbNextUnlabeled_
    pbGoSelected_
    pbComputeGT_
    pbUpdate_
    pbs_  % [1x4] array of the buttons above, for bulk enable/disable
    listener_ = cell(0,1)  % Labeler listeners owned by this controller

    % State formerly stored in guidata
    tbl_  % combined MFTable of suggestions + labeled
    err_  % [height(tbl_)] GT error per row
    hasLbl_  % [height(tbl_)] logical, whether each row is labeled
    iMovUn_  % MovieIndex array of GT movies
  end  % properties

  methods
    function obj = GTManagerController(parent, labeler)
      % Build the window and wire up Labeler listeners.
      assert(isa(labeler, 'Labeler')) ;
      obj.parent_ = parent ;
      obj.labeler_ = labeler ;

      obj.createGui_() ;

      % Listeners for table maintenance.
      obj.listener_{end+1,1} = addlistener(labeler, 'gtSuggUpdated', @(s,e)(obj.updateAll_())) ;
      obj.listener_{end+1,1} = addlistener(labeler, 'gtSuggMFTableLbledUpdated', @(s,e)(obj.updateAll_())) ;
      % Listeners for table row selection.
      obj.listener_{end+1,1} = addlistener(labeler, 'newMovie', @(s,e)(obj.currMovFrmTgtChanged_())) ;
      obj.listener_{end+1,1} = addlistener(labeler, 'didSetCurrTarget', @(s,e)(obj.currMovFrmTgtChanged_())) ;
      obj.listener_{end+1,1} = addlistener(labeler, 'gtResUpdated', @(s,e)(obj.gtResUpdated_())) ;
      obj.listener_{end+1,1} = addlistener(labeler, 'updateAfterCurrentFrameSet', @(s,e)(obj.currMovFrmTgtChanged_())) ;

      obj.updateAll_() ;
      set(obj.hFig, 'Visible', 'on') ;

      parentFig = obj.parentMainFigure_() ;
      if isscalar(parentFig) && ishghandle(parentFig)
        centerfig(obj.hFig, parentFig) ;
      end
    end  % function

    function delete(obj)
      % Tear down the Labeler listeners and the figure.
      for i = 1:numel(obj.listener_)
        o = obj.listener_{i} ;
        if isvalid(o)
          delete(o) ;
        end
      end
      obj.listener_ = cell(0,1) ;
      deleteValidGraphicsHandles(obj.hFig) ;
      obj.hFig = [] ;
    end  % function

    function createGui_(obj)
      % Build the window figure and its widgets programmatically.  The figure
      % starts hidden; the constructor makes it visible after the first update.
      hFig = uifigure('Units', 'pixels', 'Position', [951,1400,733,733], ...
                      'Name', 'Groundtruth Navigator', 'Visible', 'off') ;
      obj.hFig = hFig ;
      hFig.CloseRequestFcn = @(s,e)(obj.closeRequested_()) ;

      obj.gl_ = uigridlayout(hFig, [4,1], 'RowHeight', {'1x','1x',40}, 'tag', 'gl') ;

      obj.tblGTMovie_ = uitable(obj.gl_, ...
        'ColumnName', {'','Movie','N to Label','N Labeled'}, ...
        'RowName', {}, ...
        'ColumnWidth', {35,'1x',100,100}, ...
        'tag', 'tblGTMovie', ...
        'SelectionType', 'row', 'Multiselect', 'off', ...
        'CellSelectionCallback', @(src,evt)(obj.cellSelectionTblGTMovieActuated_())) ;

      columnnames = obj.getTblFrameColumnNames_() ;
      obj.tblFrame_ = uitable(obj.gl_, ...
        'ColumnName', columnnames, 'tag', 'tblFrame', ...
        'SelectionType', 'row', 'Multiselect', 'off', ...
        'DoubleClickedFcn', @(src,evt)(obj.doubleClickTblFrameActuated_()), ...
        'ColumnSortable', true(1,numel(columnnames))) ;

      obj.gl_buttons_ = uigridlayout(obj.gl_, [1,4], 'Padding', [0,0,0,0], 'tag', 'gl_buttons') ;
      obj.pbNextUnlabeled_ = uibutton(obj.gl_buttons_, 'Text', 'Next Unlabeled', 'tag', 'pbNextUnlabeled', ...
        'ButtonPushedFcn', @(src,evt)(obj.pbNextUnlabeledActuated_()), 'Enable', 'off') ;
      obj.pbGoSelected_ = uibutton(obj.gl_buttons_, 'Text', 'Go to Selected', 'tag', 'pbGoSelected', ...
        'ButtonPushedFcn', @(src,evt)(obj.pbGoSelectedActuated_()), 'Enable', 'off') ;
      obj.pbComputeGT_ = uibutton(obj.gl_buttons_, 'Text', 'Compute Accuracy', 'tag', 'pbComputeGT', ...
        'ButtonPushedFcn', @(src,evt)(obj.pbComputeGTActuated_()), 'Enable', 'off') ;
      obj.pbUpdate_ = uibutton(obj.gl_buttons_, 'Text', 'Update', 'tag', 'pbUpdate', ...
        'ButtonPushedFcn', @(src,evt)(obj.pbUpdateActuated_()), 'Enable', 'on') ;
      obj.pbs_ = [obj.pbNextUnlabeled_, obj.pbGoSelected_, obj.pbComputeGT_, obj.pbUpdate_] ;
      set(hFig, 'MenuBar', 'None') ;

      menu_get_gt_frames = uimenu('Tag', 'menu_get_gt_frames', 'Text', 'To-Label List', 'Parent', hFig) ;
      uimenu('Parent', menu_get_gt_frames, ...
        'MenuSelectedFcn', @(src,evt)(obj.menuGtframesSuggestActuated_()), ...
        'Label', 'Randomly select to-label list...', ...
        'Tag', 'menu_gtframes_suggest', 'Checked', 'off', 'Visible', 'on') ;
      uimenu('Parent', menu_get_gt_frames, ...
        'MenuSelectedFcn', @(src,evt)(obj.menuGtframesSetlabeledActuated_()), ...
        'Label', 'Set to-label list to current groundtruth labels', ...
        'Tag', 'menu_gtframes_setlabeled', 'Checked', 'off', 'Visible', 'on') ;
      uimenu('Parent', menu_get_gt_frames, ...
        'MenuSelectedFcn', @(src,evt)(obj.menuGtframesLoadActuated_()), ...
        'Label', 'Load to-label list from file...', ...
        'Tag', 'menu_gtframes_load', 'Checked', 'off', 'Visible', 'on') ;
    end  % function

    function columnnames = getTblFrameColumnNames_(obj)
      if obj.labeler_.projectHasTrx
        columnnames = {'Frame','Target','Labeled','Error'} ;
      else
        columnnames = {'Frame','Has Labels','Error'} ;
      end
    end  % function

    function updateTblFrame_(obj, fn)
      % Refresh the frame/target table for the selected movie.  With fn given,
      % only the named column is updated (currently no caller passes fn).
      rows = obj.tblGTMovie_.Selection ;
      columnnames = obj.getTblFrameColumnNames_() ;
      obj.tblFrame_.ColumnName = columnnames ;
      obj.tblFrame_.ColumnSortable = true(1,numel(columnnames)) ;

      doupdate = exist('fn', 'var') ;

      if isempty(rows)
        obj.tblFrame_.Data = cell(0,numel(columnnames)) ;
      else
        row = rows(1) ;
        imov = obj.iMovUn_(row) ;
        idx = obj.tbl_.mov == imov ;
        if doupdate
          data = obj.tblFrame_.Data ;
        else
          data = cell(nnz(idx),numel(columnnames)) ;
        end
        if ~doupdate || strcmpi(fn,'frm')
          data(:,1) = num2cell(obj.tbl_.frm(idx)) ;
        end
        if obj.labeler_.projectHasTrx && (~doupdate || strcmpi(fn,'iTgt'))
          data(:,2) = num2cell(obj.tbl_.iTgt(idx)) ;
        end
        if ~doupdate || strcmp(fn,'hasLbl')
          col = double(obj.labeler_.projectHasTrx) + 2 ;
          data(:,col) = num2cell(obj.hasLbl_(idx)) ;
        end
        if ~doupdate || strcmp(fn,'err')
          col = double(obj.labeler_.projectHasTrx) + 3 ;
          if any(~isnan(obj.err_))
            data(:,col) = num2cell(obj.err_(idx)) ;
          else
            data(:,col) = cell(size(data,1),1) ;
          end
        end
        obj.tblFrame_.Data = data ;
        if isempty(obj.tblFrame_.Selection) && ~isempty(data)
          setTableSelection_(obj.tblFrame_,1) ;
        end
      end
    end  % function

    function updateAll_(obj)
      % Rebuild both tables and the button enabled state from the Labeler.
      parentFig = obj.parentMainFigure_() ;
      if ~(isscalar(parentFig) && ishghandle(parentFig))
        % Sometimes an update listener fires very early, before the parent's
        % main figure exists.
        return
      end
      lObj = obj.labeler_ ;
      tbl_sugg = lObj.gtSuggMFTable ;
      tbl_label = lObj.labelGetMFTableLabeled('useTrain',0,'mftonly',true) ;
      obj.tbl_ = unique([tbl_sugg;tbl_label],'rows') ;

      obj.err_ = gtManagerGetGTErr_(obj.tbl_,lObj) ;
      obj.hasLbl_ = lObj.getIsLabeledGT(obj.tbl_) ;
      iMovUnAbs = (1:lObj.nmoviesGT)' ;
      obj.iMovUn_ = MovieIndex(iMovUnAbs,true) ;

      % replace .mov with strings
      iMov = obj.tbl_.mov ;
      iMovUnLabeledCnt = arrayfun(@(x)nnz(x==iMov&obj.hasLbl_),obj.iMovUn_) ;
      iMovUnToLabelCnt = arrayfun(@(x)nnz(x==iMov&~obj.hasLbl_),obj.iMovUn_) ;
      movStrsUn = lObj.getMovieFilesAllFullMovIdx(obj.iMovUn_) ;

      movTableData = [num2cell(iMovUnAbs(:)),movStrsUn(:,1),num2cell(iMovUnToLabelCnt(:)),num2cell(iMovUnLabeledCnt)] ;
      obj.tblGTMovie_.Data = movTableData ;

      obj.updateTblFrame_() ;

      pbEnable = onIff(~isempty(obj.tbl_)) ;
      set(obj.pbs_,'Enable',pbEnable) ;
    end  % function

    function cellSelectionTblGTMovieActuated_(obj)
      obj.updateTblFrame_() ;
    end  % function

    function gtResUpdated_(obj)
      if ~(isscalar(obj.labeler_) && isa(obj.labeler_,'Labeler'))
        return
      end
      obj.updateAll_() ;
    end  % function

    function currMovFrmTgtChanged_(obj)
      lObj = obj.labeler_ ;
      if lObj.isinit || ~lObj.hasMovie || ~lObj.gtIsGTMode
        return
      end
      mIdx = lObj.currMovIdx ;
      frm = lObj.currFrame ;
      iTgt = lObj.currTarget ;
      newrow = find(obj.iMovUn_==mIdx) ;
      oldrow = obj.tblGTMovie_.Selection ;
      if ~isequal(newrow,oldrow)
        setTableSelection_(obj.tblGTMovie_,newrow) ;
        obj.updateTblFrame_() ;
      end

      data = obj.tblFrame_.Data ;
      if lObj.projectHasTrx
        newrow = find(cell2mat(data(:,1))==frm & cell2mat(data(:,2))==iTgt) ;
      else
        newrow = find(cell2mat(data(:,1))==frm) ;
      end
      setTableSelection_(obj.tblFrame_,newrow) ;
    end  % function

    function doubleClickTblFrameActuated_(obj)
      lObj = obj.labeler_ ;
      if ~lObj.gtIsGTMode
        warningNoTrace('GTManager:nav', ...
          'Nagivation via GT Manager is disabled. Labeler is not in GT mode.') ;
        return
      end
      [mov,ft] = obj.getMFT_() ;
      if numel(mov) ~= 1 || size(ft,1) ~= 1
        return
      end
      gtManagerNavToMFT_(lObj,mov(1),ft(1,:)) ;
    end  % function

    function [movrow,ftrow] = getSelection_(obj)
      if isempty(obj.tblFrame_.Selection)
        ftrow = [] ;
      else
        ftrow = unique(obj.tblFrame_.Selection(:,1)) ;
      end
      if isempty(obj.tblGTMovie_.Selection)
        movrow = [] ;
      else
        movrow = unique(obj.tblGTMovie_.Selection(:,1)) ;
      end
    end  % function

    function [mov,ft] = getMFT_(obj, movrow, ftrow)
      lObj = obj.labeler_ ;
      if ~exist('movrow', 'var')
        [movrow,ftrow] = obj.getSelection_() ;
      end
      ft = double(cell2mat(obj.tblFrame_.Data(ftrow,1))) ;
      if lObj.projectHasTrx
        ft = [ft,double(cell2mat(obj.tblFrame_.Data(ftrow,2)))] ;
      end
      mov = obj.iMovUn_(movrow) ;
    end  % function

    function menuGtframesSuggestActuated_(obj)
      LabelerGT.generateSuggestionsUI(obj.labeler_) ;
      obj.updateAll_() ;
    end  % function

    function menuGtframesSetlabeledActuated_(obj)
      LabelerGT.setSuggestionsToLabeledUI(obj.labeler_) ;
      obj.updateAll_() ;
    end  % function

    function menuGtframesLoadActuated_(obj)
      LabelerGT.loadSuggestionsUI(obj.labeler_) ;
      obj.updateAll_() ;
    end  % function

    function pbNextUnlabeledActuated_(obj)
      % todo, use table sorting order
      lObj = obj.labeler_ ;
      if ~any(obj.hasLbl_)
        msgbox('No more unlabeled frames.','','modal') ;
      end

      [movrow0,ftrow] = obj.getSelection_() ;
      movrow = movrow0 ;
      if isempty(movrow)
        movrow = 1 ;
        ftrow = 0 ;
      end
      if isempty(ftrow)
        ftrow = 0 ;
      end
      iRow = [] ;
      for movrow = movrow:numel(obj.iMovUn_)
        mov = obj.iMovUn_(movrow) ;
        idx = obj.tbl_.mov == mov ;
        tfUnlbled = ~obj.hasLbl_(idx) ;
        iRow = find(tfUnlbled(ftrow+1:end),1) ;
        if ~isempty(iRow)
          iRow = iRow + ftrow ;
          break
        end
        ftrow = 0 ;
      end
      if isempty(iRow)
        msgbox('No more unlabeled frames.') ;
      else
        if ~isequal(movrow,movrow0)
          setTableSelection_(obj.tblGTMovie_,movrow) ;
          obj.updateAll_() ;
        end
        [mov,ft] = obj.getMFT_(movrow,iRow) ;
        setTableSelection_(obj.tblFrame_,iRow) ;
        gtManagerNavToMFT_(lObj,mov,ft) ;
      end
    end  % function

    function pbGoSelectedActuated_(obj)
      % Switch to selected row (mov/frm/tgt)
      lObj = obj.labeler_ ;
      if ~lObj.gtIsGTMode
        warningNoTrace('GTManager:nav', ...
          'Nagivation via GT Manager is disabled. Labeler is not in GT mode.') ;
        return
      end
      [mov,ft] = obj.getMFT_() ;
      if isempty(mov)
        msgbox('Please select a row in each table.','No movie selected') ;
        return
      end
      if isempty(ft)
        msgbox('Please select a row in each table.','No frame/target selected') ;
        return
      end
      gtManagerNavToMFT_(lObj,mov(1),ft(1,:)) ;
    end  % function

    function pbComputeGTActuated_(obj)
      lObj = obj.labeler_ ;
      controller = obj.parent_ ;
      if isempty(controller)
        whichlabels = 'all' ;
      else
        response = controller.askAboutUnrequestedGTLabelsIfNeeded_() ;
        if strcmp(response, 'cancel')
          return
        end
        whichlabels = response ;
      end
      lObj.gtComputeGTPerformance('whichlabels',whichlabels) ;
      obj.updateAll_() ;
    end  % function

    function pbUpdateActuated_(obj)
      obj.updateAll_() ;
    end  % function

    function closeRequested_(obj)
      delete(obj) ;
    end  % function

    function fig = parentMainFigure_(obj)
      % The parent controller's main figure, or an empty gobject if there is no
      % (scalar) parent.
      parent = obj.parent_ ;
      if isempty(parent) || ~isscalar(parent)
        fig = gobjects(1,0) ;
      else
        fig = parent.mainFigure_ ;
      end
    end  % function
  end  % methods
end  % classdef

function setTableSelection_(uitbl,row)
  % Set a uitable's row selection and scroll it into view (no-op if unchanged).
  if isequal(uitbl.Selection,row)
    return
  end
  uitbl.Selection = row ;
  if ~isempty(row)
    scroll(uitbl,"row",row) ;
  end
end  % function

function err = gtManagerGetGTErr_(tblSugg,lObj)
  % Get computed GT results/err for the given suggestion table.
  n = height(tblSugg) ;
  err = nan(n,1) ;
  tblRes = lObj.gtTblRes ;
  if ~isempty(tblRes)
    [tf,loc] = tblismember(tblSugg,tblRes,MFTable.FLDSID) ;
    err(tf) = tblRes.meanL2err(loc(tf)) ;
  end
end  % function

function gtManagerNavToMFT_(lObj,mov,ft)
  % Navigate the Labeler to the given movie and frame(/target).
  iMov = mov.get() ;
  if iMov~=lObj.currMovie
    lObj.movieSet(iMov) ;
  end
  if numel(ft) > 1
    itgt = ft(2) ;
    lObj.setFrameAndTarget(ft(1),itgt) ;
  else
    lObj.setFrame(ft(1)) ;
  end
end  % function
