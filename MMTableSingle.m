classdef MMTableSingle < MovieManagerTable
  
  properties
    jtable
  end
  
  methods
    
    function obj = MMTableSingle(hMM,hParent,position,cbkSelectMovie)
      obj@MovieManagerTable(hMM,hParent,position,cbkSelectMovie);
      
      jt = uiextras.jTable.Table(...
        'parent',hParent,...
        'Position',position,...
        'SelectionMode','discontiguous',...
        'ColumnName',{'Movie' 'Has Labels'},...
        'ColumnPreferredWidth',[600 250],...
        'Editable','off');
      jt.MouseClickedCallback = @(src,evt)obj.cbkClickedDefault(src,evt);      
      obj.jtable = jt;
    end
    
    function updateMovieData(obj,movNames,movsHaveLbls)
      assert(size(movNames,1)==numel(movsHaveLbls));
      dat = [movNames num2cell(movsHaveLbls)];
      
      jt = obj.jtable;
      if ~isequal(dat,jt.Data)
        jt.Data = dat;
      end
    end
  
    function updateSelectedMovie(obj,imov)
      jt = obj.jtable;
      tblnrows = size(jt.Data,1);
      if imov>0 && imov<=tblnrows
        jt.SelectedRows = imov;
      else
        jt.SelectedRows = [];
      end
    end

    function imovs = getSelectedMovies(obj)
      % AL20160630: IMPORTANT: currently CANNOT sort table by columns
      jt = obj.jtable;
      selRow = jt.SelectedRows;
      imovs = sort(selRow);
    end
        
  end
    
end