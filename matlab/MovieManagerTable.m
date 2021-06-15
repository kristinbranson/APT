classdef MovieManagerTable < handle
  
  properties (Constant)
    JTABLEPROPS_NOTRX = {'ColumnName',{'Movie' 'Num Labels'},...
                         'ColumnPreferredWidth',[600 250],...
                         'ColumnFormat',{'char' 'integer'}};
    JTABLEPROPS_TRX = {'ColumnName',{'Movie' 'Trx' 'Num Labels'},...
                       'ColumnFormat',{'char' 'char' 'integer'},...
                       'ColumnPreferredWidth',[550 200 100]};
  end
  properties
    hParent
    cbkSelectMovie
    
    trxShown % scalar logical. If true, table displays trxfiles
  end
  
  events
    tableClicked
  end

  methods (Abstract)
        
    % Update movienames/flags
    %
    % movNames: [nMovxnView] cellstr. Movie files
    % trxNames: [nMovxnView] cellstr. trx files
    % movsHaveLbls: [nMov x 1] logical.
    updateMovieData(obj,movNames,trxNames,movsHaveLbls)
  
    % Update currently selected row
    %
    % imov: row index into table. Currently, tables are expected never to
    % resort by row. For multiview projects, imov is the movie SET.
    updateSelectedMovie(obj,imov)
    
    % For multiview projects, imovs are the selected movie SETS. 
    imovs = getSelectedMovies(obj)
    
  end
  
  methods (Static)
    function obj = create(nMovsPerSet,hParent,position,cbkSelectMovie)
      switch nMovsPerSet
        case 1
          obj = MMTableSingle(hParent,position,cbkSelectMovie);
        otherwise
          obj = MMTableMulti(nMovsPerSet,hParent,position,cbkSelectMovie);
      end
    end
  end
  
  methods
    
    function obj = MovieManagerTable(hParent,position,cbkSelectMovie)
      % Create/initialize table.
      %
      % hParent: handle to parent of new table
      % position: [4] position vec (pixels)
      % cbkSelectMovie: function handle with sig 
      %   cbkSelectMovie(iMovSet). This is the only message that can be 
      %   sent from MMTable to MM.
      
      assert(ishandle(hParent));
      assert(isa(cbkSelectMovie,'function_handle'));
      
      obj.hParent = hParent;
      obj.cbkSelectMovie = cbkSelectMovie;
    end
    
    function delete(obj)
      obj.hParent = [];
      obj.cbkSelectMovie = [];
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
      
      obj.notify('tableClicked');      
    end    
  end
  
end