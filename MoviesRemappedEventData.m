classdef MoviesRemappedEventData < event.EventData
  % Event data for moviesRemapped event which supports movie removal and
  % reordering (but not addition)
  
  properties
    % containers.Map. mIdxOrig2New(mIdxOrig) gives mIdxNew, the new/updated 
    % movie index for mIdxOrig. If movie mIdxOrig is no longer present, 
    % mIdxNew will equal 0.
    %
    % Here mIdxOrig/mIdxNew are int32s. Conceptually they are MovieIndexes,
    % but as of 2017b containers.Map does not support arbitrary classes.
    mIdxOrig2New
    
    % MovieIndex vector. Original movies removed that had labels. 
    % Conceptually this falls a bit outside of movie-remapping, but
    % piggyback here for convenience
    mIdxRmedHadLbls 
  end
  
  methods
  
    function obj = MoviesRemappedEventData(m,nMovOrig,nMovOrigGT,...
        mIdxRemovedHadLbls)
      assert(isa(m,'containers.Map'));      
      keysOrig = num2cell(int32(1:nMovOrig));
      keysOrigGT = num2cell(int32(-1:-1:-nMovOrigGT));
      assert(all(m.isKey(keysOrig)));
      assert(all(m.isKey(keysOrigGT)));
      
      obj.mIdxOrig2New = m;
      obj.mIdxRmedHadLbls = mIdxRemovedHadLbls;
    end
    
  end
  
  methods (Static) 
    
    function obj = moviesReorderedEventData(p,nMovOrigReg,nMovOrigGT)
      % Convenience/Factory ctor for regular-movies reordered
      %
      % p: permutation of 1:nMovOrigReg
      % nMovOrigReg: original number of regular movies
      % nMovOrigGT: " GT movies
      
      p = p(:)';
      one2nmov = 1:nMovOrigReg;
      assert(isequal(sort(p),one2nmov));
      
      % p(1)->1
      % p(2)->2
      % ...
      % p(n)->n
      origIdxs = [p -1:-1:-nMovOrigGT];
      newIdxs = [one2nmov -1:-1:-nMovOrigGT];
      m = containers.Map(int32(origIdxs),int32(newIdxs));
      mIdxEmpty = MovieIndex.empty(0,1);
      obj = MoviesRemappedEventData(m,nMovOrigReg,nMovOrigGT,mIdxEmpty);
    end
        
    function obj = movieRemovedEventData(mIdx,nMovOrigReg,nMovOrigGT,...
        mIdxHadLbls)
      % Convenience/Factory ctor for movie removed case
      %
      % mIdx: scalar MovieIndex being removed
      % nMovOrigReg: original number of regular movies
      % nMovOrigGT: " GT movies
      % mIdxHadLbls: scalar logical
      
      assert(isscalar(mIdx) && isa(mIdx,'MovieIndex'));
      assert(nMovOrigReg>=0);
      assert(nMovOrigGT>=0);
      
      mIdx = int32(mIdx);
      origIdxs = [1:nMovOrigReg -1:-1:-nMovOrigGT];
      if mIdx>0
        newIdxs = [1:mIdx-1 0 mIdx:nMovOrigReg-1 ...
                  -1:-1:-nMovOrigGT];
      else
        newIdxs = [1:nMovOrigReg ...
                   -1:-1:mIdx+1 0 mIdx:-1:-nMovOrigGT+1];
      end
      m = containers.Map(int32(origIdxs),int32(newIdxs));
      if mIdxHadLbls
        mIdxRmedHadLbls = mIdx;
      else
        mIdxRmedHadLbls = MovieIndex.empty(0,1);
      end
      obj = MoviesRemappedEventData(m,nMovOrigReg,nMovOrigGT,mIdxRmedHadLbls);
    end
    
  end
end