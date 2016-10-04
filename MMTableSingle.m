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
      jt.MouseClickedCallback = @(src,evt)cbkClicked(obj,src,evt);      
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
    
    function cbkClicked(obj,src,evt)
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
        jt = obj.jtable;
        selRow = jt.SelectedRows;
        if numel(selRow)>1
          warning('MMTableSingle:sel',...
            'Multiple movies selected; switching to first selection.');
          selRow = selRow(1);
        end
        obj.cbkSelectMovie(selRow);
      end
    end
        
  end
    
end