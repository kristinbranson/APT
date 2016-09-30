classdef MovieManagerTable < handle
  
  properties
    hMM
    hParent
    cbkSelectMovie
  end

  methods (Abstract)
        
    % Update movienames/flags
    %
    % movNames: [nMovxnView] cellstr. Movies, macros allowed
    % movsHaveLbls: [nMov x 1] logical.
    updateMovieData(obj,movNames,movsHaveLbls)
  
    % Update currently selected row
    %
    % imov: row index into table. Currently, tables are expected never to
    % resort by row.
    updateSelectedMovie(obj,imov)
    
    imovs = getSelectedMovies(obj)
    
  end
  
  methods (Static)
    function obj = create(nView,hMM,hParent,position,cbkSelectMovie)
      switch nView
        case 1
          obj = MMTableSingle(hMM,hParent,position,cbkSelectMovie);
        otherwise
          obj = MMTableMulti(hMM,hParent,position,cbkSelectMovie);
      end
    end
  end
  
  methods
    
    function obj = MovieManagerTable(hMM,hParent,position,cbkSelectMovie)
      % Create/initialize table.
      %
      % hMM: MovieManager handle
      % hParent: handle to parent of new table
      % position: [4] position vec (pixels)
      % cbkSelectMovie: function handle with sig 
      %   cbkSelectMovie(movname). This is the only message that can be sent
      %   from MMTable to MM.
      
      assert(ishandle(hMM));
      assert(ishandle(hParent));
      assert(isa(cbkSelectMovie,'function_handle'));
      
      obj.hMM = hMM;
      obj.hParent = hParent;
      obj.cbkSelectMovie = cbkSelectMovie;
    end

  end
    
end