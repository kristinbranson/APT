classdef LabelCoreSeqAddModel < LabelCoreSeqModel
% Sequential adding of new landmarks - model
%
% Extends LabelCoreSeqModel for the case where some landmarks already
% exist (nold) and only new ones (nadd) are being added. Old landmarks
% are frozen and cannot be moved.

  properties (Transient)
    nadd_                   % number of landmarks being added
    nold_                   % number of landmarks to leave frozen
  end

  methods

    function obj = LabelCoreSeqAddModel(labeler)
      % Construct a LabelCoreSeqAddModel.
      obj = obj@LabelCoreSeqModel(labeler) ;
    end  % function

    function initHook(obj)
      % Initialize SeqAdd-specific state.
      obj.nadd_ = obj.labeler_.nLabelPointsAdd ;
      obj.nold_ = obj.nPts_ - obj.nadd_ ;
      obj.kpfIPtFor1Key_ = obj.nold_ + 1 ;
      obj.state_ = LabelState.ADJUST ;
    end  % function

  end  % methods

  %% State transition hooks (override parent)
  methods

    function clearLabels(obj)
      % Clear only the new points (not old frozen ones).
      obj.beginAdd([], true) ;
    end  % function

    function acceptLabels(obj)
      % Accept labels for the current frame/target.
      obj.beginAccepted(true) ;
      obj.labeler_.restorePrevAxesMode() ;
    end  % function

    function unAcceptLabels(obj)
      % Go to the next frame to label.
      if ~isempty(obj.nexttbl_)
        obj.goToNextTable() ;
      else
        obj.goToNextUnlabeled() ;
      end
    end  % function

    function setNextTable(obj, tbl)
      % Set table and navigate to next partially labeled frame.
      obj.setNextTable@LabelCoreModel(tbl) ;
      obj.goToNextTable() ;
    end  % function

  end  % methods

  %% Internal state transitions
  methods

    function newFrameTarget(obj, iFrm, iTgt)
      % React to new frame or target. Handle partially-labeled frames.

      [tflabeled, lpos, lpostag] = obj.labeler_.labelPosIsPtLabeled(iFrm, iTgt) ;
      obj.nPtsLabeled_ = nnz(tflabeled) ;

      obj.setLabelCoords(lpos, 'lblTags', lpostag) ;
      obj.iPtMove_ = nan ;

      if obj.nPtsLabeled_ < obj.nold_
        obj.beginAccepted(false) ;
      elseif obj.nPtsLabeled_ == obj.nold_
        obj.beginAdd(lpos, false) ;
      else
        obj.beginAccepted(false) ;
      end
    end  % function

    function beginAdd(obj, lpos, tfClearLabels)
      % Enter Label state for adding new points, keeping old points frozen.

      if ~isempty(lpos)
        obj.setLabelCoords(lpos) ;
      end
      obj.nPtsLabeled_ = obj.nold_ ;
      obj.iPtMove_ = nan ;
      obj.tfOcc_(obj.nold_+1:end) = false ;
      obj.tfEstOcc_(obj.nold_+1:end) = false ;
      obj.tfSel_(obj.nold_+1:end) = false ;
      if tfClearLabels
        obj.labeler_.labelPosClearPoints(obj.nold_+1:obj.nPts_) ;
        [~, lpos2, ~] = obj.labeler_.labelPosIsPtLabeled( ...
          obj.labeler_.currFrame, obj.labeler_.currTarget) ;
        obj.setLabelCoords(lpos2) ;
      end
      obj.state_ = LabelState.LABEL ;
      obj.notify('updateState') ;
    end  % function

    function beginAccepted(obj, tfSetLabelPos)
      % Enter accepted state for current frame.

      if tfSetLabelPos
        xy = obj.getLabelCoords() ;
        obj.labeler_.labelPosSet(xy) ;
        obj.setLabelPosTagFromEstOcc() ;
      end
      obj.iPtMove_ = nan ;
      obj.clearSelected() ;
      obj.state_ = LabelState.ACCEPTED ;
      obj.notify('updateState') ;
    end  % function

  end  % methods

  %% Action methods (override parent where needed)
  methods

    function labelNextPoint(obj, xy, tfAxOcc, tfShift)
      % Label the next sequential point (same as parent).
      % Difference: acceptance auto-triggers after all nPts.
      labelNextPoint@LabelCoreSeqModel(obj, xy, tfAxOcc, tfShift) ;
    end  % function

    function undoLastLabel(obj)
      % Undo the last label placement (same as parent).
      undoLastLabel@LabelCoreSeqModel(obj) ;
    end  % function

  end  % methods

  %% Navigation
  methods

    function goToNextTable(obj)
      % Navigate to the next partially-labeled frame in the table.

      labels = obj.labeler_.labelsGTaware() ;
      todo = false(size(obj.nexttbl_, 1), 1) ;
      [movnames, ~, rowidx] = unique(obj.nexttbl_.mov) ;
      for i = 1:numel(movnames)
        idxcurr = rowidx == i ;
        [tf, mov, ~, ~] = obj.labeler_.movieSetInProj(movnames{i}) ;
        if ~tf
          error('LabelCoreSeqAddModel:badNextTable', ...
                'Movie %s is not in this project', movnames{i}) ;
        end
        [frms, tgts] = Labels.isPartiallyLabeledT(labels{mov}, nan, obj.nold_) ;
        todo(idxcurr) = ismember( ...
          [obj.nexttbl_.frm(idxcurr), obj.nexttbl_.iTgt(idxcurr)], ...
          [frms(:), tgts(:)], 'rows') ;
      end

      nextj = find(todo(obj.nexti_+1:end), 1) ;
      if isempty(nextj)
        nextj = find(todo, 1) ;
        if isempty(nextj)
          labeler = obj.labeler_ ;
          labeler.dialogLaunchPad_ = struct('text', 'No partially labeled frames found within the table! You might be done!', ...
                                            'title', 'Done adding landmarks') ;
          labeler.notify('requestMessageBox') ;
          return ;
        else
          obj.nexti_ = nextj ;
        end
      else
        obj.nexti_ = obj.nexti_ + nextj ;
      end

      frm = obj.nexttbl_.frm(obj.nexti_) ;
      if ismember('iTgt', obj.nexttbl_.Properties.VariableNames)
        tgt = obj.nexttbl_.iTgt(obj.nexti_) ;
      end
      movname = obj.nexttbl_.mov{obj.nexti_} ;
      [tf, mov, ~, ~] = obj.labeler_.movieSetInProj(movname) ;
      if ~tf
        error('LabelCoreSeqAddModel:badNextTable', ...
              'Movie %s is not in this project', movname) ;
      end
      if obj.labeler_.currMovie ~= mov
        obj.labeler_.movieSet(mov) ;
      end
      if obj.labeler_.currTarget ~= tgt
        obj.labeler_.setFrameAndTarget(frm, tgt) ;
      else
        obj.labeler_.setFrame(frm) ;
      end
    end  % function

    function goToNextUnlabeled(obj)
      % Navigate to the next partially-labeled frame in the project.
      [frm, tgt, mov] = obj.findUnlabeled() ;
      if isempty(frm)
        labeler = obj.labeler_ ;
        labeler.dialogLaunchPad_ = struct('text', 'No partially labeled frames found! You might be done!', ...
                                          'title', 'Done adding landmarks') ;
        labeler.notify('requestMessageBox') ;
        return ;
      end
      if obj.labeler_.currMovie ~= mov
        obj.labeler_.movieSet(mov) ;
      end
      if obj.labeler_.currTarget ~= tgt
        obj.labeler_.setFrameAndTarget(frm, tgt) ;
      else
        obj.labeler_.setFrame(frm) ;
      end
    end  % function

    function [frm, tgt, mov] = findUnlabeled(obj)
      % Find the next partially-labeled frame in the project.
      labels = obj.labeler_.labelsGTaware() ;
      currtgt = obj.labeler_.currTarget ;
      currmov = obj.labeler_.currMovie ;
      currfrm = obj.labeler_.currFrame ;
      frms = Labels.isPartiallyLabeledT(labels{currmov}, currtgt, obj.nold_) ;
      if ~isempty(frms)
        i = find(frms > currfrm, 1) ;
        if isempty(i)
          frm = frms(1) ;
        else
          frm = frms(i) ;
        end
        tgt = currtgt ;
        mov = currmov ;
        return ;
      end
      [frms, tgts] = Labels.isPartiallyLabeledT(labels{currmov}, nan, obj.nold_) ;
      if ~isempty(frms)
        frm = frms(1) ;
        tgt = tgts(1) ;
        mov = currmov ;
        return ;
      end
      for mov = 1:size(labels, 1)
        [frms, tgts] = Labels.isPartiallyLabeledT(labels{mov}, nan, obj.nold_) ;
        if ~isempty(frms)
          frm = frms(1) ;
          tgt = tgts(1) ;
          return ;
        end
      end
      frm = [] ;
      tgt = [] ;
      mov = [] ;
    end  % function

  end  % methods

end  % classdef
