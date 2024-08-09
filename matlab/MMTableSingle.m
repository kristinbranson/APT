classdef MMTableSingle < MovieManagerTable
  
  properties
    jtable
  end
  
  methods
    
    function obj = MMTableSingle(hParent,position,cbkSelectMovie)
      obj@MovieManagerTable(hParent,position,cbkSelectMovie);
      
      jt = uiextras.jTable.Table(...
        'parent',hParent,...
        'Position',position,...
        'SelectionMode','discontiguous',...
        'Editable','off',...
        obj.JTABLEPROPS_NOTRX{:});
      jt.MouseClickedCallback = @(src,evt)obj.cbkClickedDefault(src,evt);      
      obj.jtable = jt;
    end
    
    function updateMovieData(obj,movNames,trxNames,movsHaveLbls)
      szassert(trxNames,size(movNames));
      assert(size(movNames,1)==numel(movsHaveLbls));
      
      tfTrx = any(cellfun(@(x)~isempty(x),trxNames));
      if tfTrx
        assert(size(trxNames,2)==1,'Expect single column.');
        dat = [num2cell(int64(1:size(movNames,1)))' movNames trxNames num2cell(int64(movsHaveLbls))];
        jtArgs = MovieManagerTable.JTABLEPROPS_TRX;
      else
        dat = [num2cell(int64(1:size(movNames,1)))' movNames num2cell(int64(movsHaveLbls))];
        jtArgs = MovieManagerTable.JTABLEPROPS_NOTRX;
      end
      
      jt = obj.jtable;
      if ~isequal(dat,jt.Data)
        set(jt,jtArgs{:},'Data',dat);
      end
    end
  
    function updateSelectedMovie(obj,imov)
      jt = obj.jtable;
      tblnrows = size(jt.Data,1);
      if imov>0 && imov<=tblnrows
        jt.SelectedRows = int32(imov);
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