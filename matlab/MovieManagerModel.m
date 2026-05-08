classdef MovieManagerModel < handle
  % State that backs the MovieManagerController view but is not
  % itself graphical.  Lives on the Labeler so it persists across
  % MMC open/close cycles, and so batch code can mutate it without
  % a controller present.

  properties
    showPathEnds_ = true
      % logical scalar.  When true, the main table's Movie/Trx
      % columns truncate from the start (showing the file-name end);
      % when false, paths are shown untruncated.
    moviesSelected_ = []
      % [nSel x 1] vector of MovieIndices currently highlighted in
      % MovieManager.  GT mode ok.  Set by the MMC in response to
      % user selection.
    originalMovNames_ = {}
      % [nMov x nView] cellstr cache of the most recent untruncated
      % movie names seen by updateMovieData_.  Used to re-truncate
      % on figure resize without re-querying the model.
    originalTrxNames_ = {}
      % [nMov x nView] cellstr cache of the most recent untruncated
      % trx names.
    originalMovsHaveLbls_ = []
      % [nMov x 1] logical cache of the most recent has-labels flags.
  end

  properties (Dependent)
    showPathEnds
    moviesSelected
    originalMovNames
    originalTrxNames
    originalMovsHaveLbls
  end

  events
    didSetShowPathEnds
    didSetMoviesSelected
    didSetOriginalNames
  end

  methods
    function obj = MovieManagerModel()
      % Pure data class; no construction-time arguments.
    end  % function

    function v = get.showPathEnds(obj)
      v = obj.showPathEnds_ ;
    end  % function

    function set.showPathEnds(obj, v)
      obj.showPathEnds_ = logical(v) ;
      obj.notify('didSetShowPathEnds') ;
    end  % function

    function v = get.moviesSelected(obj)
      v = obj.moviesSelected_ ;
    end  % function

    function set.moviesSelected(obj, v)
      obj.moviesSelected_ = v ;
      obj.notify('didSetMoviesSelected') ;
    end  % function

    function v = get.originalMovNames(obj)
      v = obj.originalMovNames_ ;
    end  % function

    function v = get.originalTrxNames(obj)
      v = obj.originalTrxNames_ ;
    end  % function

    function v = get.originalMovsHaveLbls(obj)
      v = obj.originalMovsHaveLbls_ ;
    end  % function

    function setOriginalNames(obj, movNames, trxNames, movsHaveLbls)
      % Atomic update of all three caches with a single notification.
      obj.originalMovNames_ = movNames ;
      obj.originalTrxNames_ = trxNames ;
      obj.originalMovsHaveLbls_ = movsHaveLbls ;
      obj.notify('didSetOriginalNames') ;
    end  % function
  end  % methods
end  % classdef
