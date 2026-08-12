classdef TrkInfoController < handle
  % Controller for the "Track Info" window, which summarizes tracklets for the
  % current movie of a multi-animal project and lets the user navigate to
  % tracklet starts/ends and breaks.
  %
  % This replaces the bare uifigure function TrkInfoUI.m: state and widget
  % handles move from guidata into instance properties.  The old function's
  % find-or-create singleton behavior is now handled by the caller
  % (LabelerController), which holds one controller and re-raises it via
  % raiseAndSyncToCurrentMovie_().

  properties (Transient)
    hFig  % the window figure
    parent_  % a LabelerController (provides tvTrkPred_)
    labeler_  % a Labeler
    mov_tbl_  % movie list uitable
    tbl_  % tracklet summary uitable
    sf_btn_
    ef_btn_
    prev_btn_
    next_btn_

    % State formerly stored in guidata
    sf_ = []  % [ntracklet] start frames of current movie's tracklets
    ef_ = []  % [ntracklet] end frames
    breaks_ = {}
    top_links_ = {}
    data_ = {}
    has_data_ = false
    curtrk_ = []  % currently-selected tracklet, or empty
    curmov_  % currently-shown movie index
    trk_ = {}  % current TrkFile, or {} if none
  end  % properties

  methods
    function obj = TrkInfoController(parent, labeler)
      % Build the window for the given multi-animal Labeler.
      assert(labeler.maIsMA, 'UI is functional only for multi-animal projects') ;
      obj.parent_ = parent ;
      obj.labeler_ = labeler ;
      obj.curmov_ = labeler.currMovie ;
      obj.createGui_() ;
      obj.updateMovie_() ;
    end  % function

    function delete(obj)
      deleteValidGraphicsHandles(obj.hFig) ;
      obj.hFig = [] ;
    end  % function

    function createGui_(obj)
      % Build the window figure and its widgets programmatically.
      f = uifigure('Units', 'pixel', 'Position', [250,250,1000,800], ...
                   'tag', 'TrkInfoUI', 'Name', 'Track Info') ;
      obj.hFig = f ;

      nr = 3 ;
      nc = 2 ;
      gl = uigridlayout(f, [nr nc]) ;
      gl.RowHeight = {150, '1x', 50} ;
      gl.ColumnWidth = {'1x', 100} ;

      mov_list = obj.labeler_.movieFilesAllFullGTaware ;
      mov_tbl = uitable(gl, 'Data', mov_list) ;
      mov_tbl.Layout.Row = 1 ;
      mov_tbl.Layout.Column = 1 ;
      mov_tbl.ColumnName = {'Movie'} ;
      mov_tbl.ColumnWidth = {'1x'} ;
      mov_tbl.CellSelectionCallback = @(s,e)(obj.cellClickMovActuated_(e)) ;
      obj.mov_tbl_ = mov_tbl ;

      tbl = uitable(gl) ;
      tbl.Layout.Row = 2 ;
      tbl.Layout.Column = [1 nc] ;
      tbl.CellSelectionCallback = @(s,e)(obj.cellClickTblActuated_(e)) ;
      obj.tbl_ = tbl ;

      gl_btm = uigridlayout(gl, [1 4]) ;
      gl_btm.Layout.Row = 3 ;
      gl_btm.Layout.Column = [1 nc] ;

      obj.sf_btn_ = uibutton(gl_btm, 'Text', 'First Frame', 'ButtonPushedFcn', @(s,e)(obj.sfBtnActuated_())) ;
      obj.sf_btn_.Layout.Row = 1 ;
      obj.sf_btn_.Layout.Column = 1 ;
      obj.ef_btn_ = uibutton(gl_btm, 'Text', 'End Frame', 'ButtonPushedFcn', @(s,e)(obj.efBtnActuated_())) ;
      obj.ef_btn_.Layout.Row = 1 ;
      obj.ef_btn_.Layout.Column = 2 ;
      obj.prev_btn_ = uibutton(gl_btm, 'Text', 'Prev break', 'ButtonPushedFcn', @(s,e)(obj.prevBtnActuated_())) ;
      obj.prev_btn_.Layout.Row = 1 ;
      obj.prev_btn_.Layout.Column = 3 ;
      obj.next_btn_ = uibutton(gl_btm, 'Text', 'Next break', 'ButtonPushedFcn', @(s,e)(obj.nextBtnActuated_())) ;
      obj.next_btn_.Layout.Row = 1 ;
      obj.next_btn_.Layout.Column = 4 ;

      obj.mov_tbl_.Selection = [obj.curmov_ 1] ;
    end  % function

    function raiseAndSyncToCurrentMovie_(obj)
      % Bring the window to the front and, if the Labeler's current movie has
      % changed, re-sync the display to it.  Mirrors the old TrkInfoUI's
      % find-existing-singleton branch.
      figure(obj.hFig) ;
      if obj.curmov_ ~= obj.labeler_.currMovie
        obj.curmov_ = obj.labeler_.currMovie ;
        obj.mov_tbl_.Selection = [obj.curmov_ 1] ;
        obj.updateMovie_() ;
      end
    end  % function

    function updateMovie_(obj)
      % Sync the tracklet table to the current movie's tracking results.
      idx = obj.curmov_ ;
      lobj = obj.labeler_ ;
      if ~isempty(lobj.tracker)
        pred_trk = lobj.tracker.getTrackingResults(MovieIndex(idx)) ;
        pred_trk = pred_trk{1} ;
      else
        pred_trk = [] ;
      end

      has_pred = ~isempty(pred_trk) && pred_trk.hasdata() ;

      if ~has_pred
        obj.tbl_.Data = {} ;
        obj.trk_ = {} ;
        obj.has_data_ = false ;
        return ;
      end

      trk = pred_trk ;
      [dat, sf, ef, breaks, top_links] = trkInfoGetData_(trk) ;
      obj.tbl_.Data = dat ;
      obj.sf_ = sf ;
      obj.ef_ = ef ;
      obj.breaks_ = breaks ;
      obj.top_links_ = top_links ;
      obj.data_ = dat ;
      obj.tbl_.ColumnSortable = true ;
      obj.trk_ = trk ;
      obj.has_data_ = true ;
    end  % function

    function cellClickMovActuated_(obj, event)
      obj.mov_tbl_.Selection = [obj.curmov_ 1] ;
      pause(0.5) ;  % allow the user time to add a second click
      if strcmpi(obj.hFig.SelectionType, 'open')
        obj.curmov_ = event.Indices(1) ;
        obj.mov_tbl_.Selection = [obj.curmov_ 1] ;
        obj.updateMovie_() ;
      end
    end  % function

    function cellClickTblActuated_(obj, event)
      pause(0.5) ;  % allow the user time to add a second click
      if strcmpi(obj.hFig.SelectionType, 'open')
        obj.switchTarget_(event.Indices(1)) ;
      end
    end  % function

    function switchTarget_(obj, tgt)
      % Navigate the Labeler to tracklet tgt (prompting for a movie switch if
      % needed) and select it in the tracking visualizer.
      if ~obj.has_data_, return ; end
      if obj.labeler_.currMovie ~= obj.curmov_
        qstr = sprintf('Switch to movie %d?', obj.curmov_) ;
        res = questdlg(qstr, 'Switch Movie') ;
        if strcmp(res, 'Yes')
          obj.labeler_.movieSet(obj.curmov_) ;
        else
          return ;
        end
      end
      lobj = obj.labeler_ ;
      sf = obj.sf_ ;
      ef = obj.ef_ ;
      trk = obj.trk_ ;
      curfr = lobj.currFrame ;
      if (sf(tgt)>curfr) || (ef(tgt)<curfr)
        lobj.setFrame(sf(tgt)) ;
      else
        haspred = trk.getPTrkFT(curfr,tgt) ;
        if ~haspred
          % set frame to the closest frame that has a prediction for the
          % current tracklet
          tdat = trk.getPTrkTgt(tgt) ;
          vfr = find(~all(isnan(tdat),[1,2])) ;
          closest = argmin(abs(vfr-curfr+sf(tgt))) ;
          lobj.setFrame(vfr(closest)) ;
        end
      end

      tvm = lobj.tracker.trkVizer ;
      if ~isempty(tvm)
        tvm.setSelectedTracklet(tgt) ;
      end
      tv = obj.parent_.tvTrkPred_ ;
      if ~isempty(tv)
        tv.centerPrimary() ;
      end
      obj.curtrk_ = tgt ;
    end  % function

    function centerPrimary_(obj)
      tv = obj.parent_.tvTrkPred_ ;
      if ~isempty(tv)
        tv.centerPrimary() ;
      end
    end  % function

    function prevBtnActuated_(obj)
      if isempty(obj.tbl_.Selection)
        return ;
      end
      curtrk = obj.tbl_.Selection(1) ;
      trk = obj.trk_ ;
      lobj = obj.labeler_ ;

      sf = trk.startframes(curtrk) ;
      ef = trk.endframes(curtrk) ;
      valid = trk.getPTrkFT(sf:ef,curtrk) ;
      [ss,ee] = get_interval_ends(valid) ;
      ee = ee-1 ;
      ss = [ss;ee] ;
      curfr = lobj.currFrame ;
      if curfr<sf
        warning('No previous breaks') ;
      elseif curfr>ef
        lobj.setFrame(ef) ;
      else
        ss(ss>=(curfr-sf+1)) = nan ;
        if all(isnan(ss))
          lobj.setFrame(sf) ;
        else
          sndx = argmax(ss) ;
          lobj.setFrame(ss(sndx)+sf-1) ;
        end
      end
      if isempty(obj.curtrk_) || (obj.curtrk_ ~= curtrk)
        obj.switchTarget_(curtrk) ;
      end
      obj.centerPrimary_() ;
    end  % function

    function nextBtnActuated_(obj)
      if isempty(obj.tbl_.Selection)
        return ;
      end
      curtrk = obj.tbl_.Selection(1) ;
      trk = obj.trk_ ;
      lobj = obj.labeler_ ;

      sf = trk.startframes(curtrk) ;
      ef = trk.endframes(curtrk) ;
      valid = trk.getPTrkFT(sf:ef,curtrk) ;
      [ss,ee] = get_interval_ends(valid) ;
      ee = ee-1 ;
      ss = [ss;ee] ;
      curfr = lobj.currFrame ;
      if curfr>ef
        warning('No next breaks') ;
      elseif curfr<sf
        lobj.setFrame(sf) ;
      else
        ss(ss<=(curfr-sf+1)) = nan ;
        if all(isnan(ss))
          lobj.setFrame(ef) ;
        else
          sndx = argmin(ss) ;
          lobj.setFrame(ss(sndx)+sf-1) ;
        end
      end
      if isempty(obj.curtrk_) || (obj.curtrk_ ~= curtrk)
        obj.switchTarget_(curtrk) ;
      end
      obj.centerPrimary_() ;
    end  % function

    function sfBtnActuated_(obj)
      if isempty(obj.tbl_.Selection)
        return ;
      end
      curtrk = obj.tbl_.Selection(1) ;
      trk = obj.trk_ ;
      lobj = obj.labeler_ ;

      sf = trk.startframes(curtrk) ;
      lobj.setFrame(sf) ;
      if isempty(obj.curtrk_) || (obj.curtrk_ ~= curtrk)
        obj.switchTarget_(curtrk) ;
      end
      obj.centerPrimary_() ;
    end  % function

    function efBtnActuated_(obj)
      if isempty(obj.tbl_.Selection)
        return ;
      end
      curtrk = obj.tbl_.Selection(1) ;
      trk = obj.trk_ ;
      lobj = obj.labeler_ ;

      ef = trk.endframes(curtrk) ;
      lobj.setFrame(ef) ;
      if isempty(obj.curtrk_) || (obj.curtrk_ ~= curtrk)
        obj.switchTarget_(curtrk) ;
      end
      obj.centerPrimary_() ;
    end  % function
  end  % methods
end  % classdef

function [tdat,sf,ef,breaks,top_links] = trkInfoGetData_(trk)
  % Summarize a TrkFile's tracklets into a table plus start/end/break info.
  n_trk = trk.ntracklets ;
  sf = trk.getStartFrame() ;
  ef = trk.getEndFrame() ;
  varNames = {'ID','N Frm','Trk Len', 'Start','End',...
    'Brks','Avg Bout Sz','Avg Brk Sz',...
    'Median Link','Max Link','90 Prc Link'} ;
  nvar = numel(varNames) ;
  tdat = table('Size',[n_trk,nvar],'VariableTypes',repmat({'double'},[1,nvar]),...
    'VariableNames',varNames) ;
  breaks = cell(1,n_trk) ;
  top_links = [] ;

  for ndx = 1:n_trk
    nfr = ef(ndx)-sf(ndx)+1 ;
    curt = trk.getPTrkTgt(ndx) ;
    valid_pred = shiftdim(~all(isnan(curt(:,1,:)),1),2) ;
    n_pred = nnz(valid_pred) ;
    [si,ei] = get_interval_ends(valid_pred) ;
    n_breaks = numel(si)-1 ;
    int_size = mean(ei-si) ;
    if n_breaks>0
      break_size = si(2:end) - ei(1:end-1) ;
      avg_break_size = mean(break_size) ;
    else
      avg_break_size = 0 ;
    end

    link = abs(curt(:,:,2:end)-curt(:,:,1:end-1)) ;
    link = nanmean(sum(link,2),1) ;
    link = shiftdim(link,2) ;
    link = link(~isnan(link)) ;
    med_link = nanmedian(link) ;
    if numel(link)>0
      max_link = nanmax(link) ;
    else
      max_link = NaN ;
    end
    link_90 = prctile(link,90) ;
    id = trk.pTrkiTgt(ndx) ;
    tdat(ndx,:) = {id,n_pred, nfr, sf(ndx), ef(ndx), n_breaks, ...
      int_size, avg_break_size,med_link,max_link,link_90} ;
    breaks{ndx} = [si,ei] ;
  end
end  % function
