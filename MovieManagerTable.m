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
    % resort by row. For multiview projects, imov is the movie SET.
    updateSelectedMovie(obj,imov)
    
    % For multiview projects, imovs are the selected movie SETS. 
    imovs = getSelectedMovies(obj)
    
  end
  
  methods (Static)
    function obj = create(nMovsPerSet,hMM,hParent,position,cbkSelectMovie)
      switch nMovsPerSet
        case 1
          obj = MMTableSingle(hMM,hParent,position,cbkSelectMovie);
        otherwise
          obj = MMTableMulti(nMovsPerSet,hMM,hParent,position,cbkSelectMovie);
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
      %   cbkSelectMovie(iMovSet). This is the only message that can be 
      %   sent from MMTable to MM.
      
      assert(ishandle(hMM));
      assert(ishandle(hParent));
      assert(isa(cbkSelectMovie,'function_handle'));
      
      obj.hMM = hMM;
      obj.hParent = hParent;
      obj.cbkSelectMovie = cbkSelectMovie;
    end

  end
  
  methods % util    
    function cbkClickedDefault(obj,src,evt)
      persistent chk
      PAUSE_DURATION_CHECK = 0.25;
      if isempty(chk)
        chk = 1;
        pause(PAUSE_DURATION_CHECK); %Add a delay to distinguish single click from a double click
        if chk==1
          % single-click; no-op
          chk = [];
        end
      else
        chk = [];
        imovs = obj.getSelectedMovies();
        if numel(imovs)>1
          warning('MovieManagerTable:sel',...
            'Multiple movies selected. Switching to first selection.');
          imovs = imovs(1);
        end
%         try
        obj.cbkSelectMovie(imovs);
%         catch ME
%           disp(ME.message);
%         end
      end
      
    end    
  end
  
end