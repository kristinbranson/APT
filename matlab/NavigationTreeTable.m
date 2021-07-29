classdef NavigationTreeTable < handle
  % TreeTable with navigation api
  %
  % Like NavigationTable but using treeTable
  
  % The "data" table .treeTblData is the underlying data model. Currently
  % the first col is always taken to be the grouping variable. For the 
  % moment this first col must be of char type. The data table is indexed 
  % by a row index iData.
  %
  % "Groups" and "sets" are synonymous.
  %
  % The treeTable is a UI representation of the data. Its groups can be 
  % expanded/collapsed. Rows of the treeTable are indexed depending on
  % which groups are currently expanded/collapsed. In general there is a
  % complicated mapping between the treeTable/UI row index and the
  % data index iData.
  
  properties
    hParent % graphics handle
    
    treeTblData % table form of treeTbl data for convenience -- treeTbl API is a bit opaque
    treeTblDataPrettyHeaders 
    treeTbl % treeTable handle
    treeTblRowObjs % [nTreeTableRowx1] cell array of row objects; generated when treeTable fully expanded
    iData2iTTExpanded1B % [nDataRowx1] index vectors. .iData2iTT(iData) gives the 1-based row index into .treeTableRowObjs
    fld2ttCol0B % containers.Map. Keys: field. value: 0-based column/field index for treeTable. grouping field NOT included
    
    fcnDataRowNaved % fcnhandle, fcnDataRowNaved(iData)
    navOnSingleClick % If true, navigate on single click; otherwise require double-click
  end
  properties (Dependent)
    isEmpty
    nData % height(.treeTblData)
    fields % cellstr of fields in current data
    groupFieldPrettyName % name of grouping field (first col in table)
    groupTreeTblRowREPat % regexp pat for grouping rows in treeTable
  end
  
  properties (Constant)
    ICONFILENAMES = {'' ...
      fullfile(matlabroot,'/toolbox/matlab/icons/file_open.png') ...
      fullfile(matlabroot,'/toolbox/matlab/icons/foldericon.gif')};
  end
  
  methods
    function tf = get.isEmpty(obj)
      tf = isempty(obj.treeTblData);
    end
    function v = get.nData(obj)
      v = size(obj.treeTblData,1);
    end
    function v = get.fields(obj)
      tblDat = obj.treeTblData;
      if isempty(tblDat)
        v = [];
      else
        v = tblDat.Properties.VariableNames;
      end
    end
    function v = get.groupFieldPrettyName(obj)
      pHdrs = obj.treeTblDataPrettyHeaders;
      if isempty(pHdrs)
        v = [];
      else
        v = pHdrs{1};
      end
    end      
    function v = get.groupTreeTblRowREPat(obj)
      v = sprintf('%s: (?<set>.+)$',obj.groupFieldPrettyName);
    end
  end
  
  methods
    
    function obj = NavigationTreeTable(hPrnt,posn,cbkNavDataRow)
      assert(isgraphics(hPrnt) && isscalar(hPrnt));
      %szassert(posn,[1 4]);
      assert(isa(cbkNavDataRow,'function_handle'));

      % don't actually create the treetable here
      obj.hParent = hPrnt;
      obj.treeTblData = [];
      obj.treeTblDataPrettyHeaders = [];
      obj.treeTbl = [];
      obj.treeTblRowObjs = [];
      obj.iData2iTTExpanded1B = [];
      obj.navOnSingleClick = false;
      obj.fcnDataRowNaved = cbkNavDataRow;
    end
    
    function delete(obj)
      if ~isempty(obj.treeTblData)
        obj.treeTblData = [];
      end
      delete(obj.treeTbl);
      if ~isempty(obj.treeTblRowObjs)
        % Throwing "Arguments must contain a char vec" on 17b. treeTable should own/cleanup these objs 
        %cellfun(@delete,obj.treeTblRowObjs);
        obj.treeTblRowObjs = [];
      end
      if ~isempty(obj.fld2ttCol0B)
        delete(obj.fld2ttCol0B)
        obj.fld2ttCol0B = [];
      end
      if ~isempty(obj.fcnDataRowNaved)
        obj.fcnDataRowNaved = [];
      end
    end
    
  end
  
  methods
    
    function tf = isfield(obj,fld)
      tblDat = obj.treeTblData;
      if isempty(tblDat)
        tf = false;
      else
        tf = tblfldscontains(tblDat,fld);
      end
    end
    
    function setData(obj,tbl,varargin)
      % tbl: table. First variable is assumed to be Set/Grouping variable.
      %
      % At the moment it is assumed in this API that the set/grouping
      % variable values are of char type, but generalizing should be 
      % straightforward.
      %
      % create the treetable here
      
      [colPreferredWidths,treeTableArgs,prettyHdrs] = myparse(varargin,...
        'colPreferredWidths',[],... % containers.Map. Keys, columns of tbl. Vals, normalized col widths. All vals sum to 1.0.
        'treeTableArgs',{},...
        'prettyHdrs',[]... % optional, cellstr of headings to use
        );

      assert(istable(tbl));
      [ndatarow,tblwidth] = size(tbl);
      assert(tblwidth>=2);
            
      NavigationTreeTable.verifyDataTable(tbl);
      tfPrettyHdrs = ~isempty(prettyHdrs);
      if tfPrettyHdrs
        assert(iscellstr(prettyHdrs) && numel(prettyHdrs)==width(tbl));
      end

      dat = table2cell(tbl);
      % insert an empty 2nd column; the 2nd column header appears in the 
      % 1st physical column
      dat = [dat(:,1) repmat({''},ndatarow,1) dat(:,2:end)];
      rawHdrs = tblflds(tbl);
      if ~tfPrettyHdrs
        prettyHdrs = rawHdrs;
      end
      prettyHdrsOrig = prettyHdrs;
      rawHdrs = rawHdrs([1 1:end]);
      prettyHdrs = prettyHdrs([1 1:end]);
%       types = arrayfun(@(x)NavigationTreeTable.col2type(dat(:,x)),...
%         1:tblwidth+1,'uni',0);
      editable = num2cell(false(1,tblwidth+1));
      
      tt = treeTable(obj.hParent,prettyHdrs,dat,...%        'ColumnTypes',types,...
        'ColumnEditable',editable,...
        'Groupable',true,...
        'IconFilenames',NavigationTreeTable.ICONFILENAMES,...
        treeTableArgs{:});
      tt.MouseClickedCallback = @(s,e)obj.cbkTableClick(s,e);
      tt.setDoubleClickEnabled(false);
      obj.treeTbl = tt;
      
      iD2TT = nan(ndatarow,1);
      ttRowObjs = cell(tt.RowCount,1);
      ctr = 0;
      for iTT0B = 0:tt.RowCount-1
        r = tt.getRowAt(iTT0B);
        ttRowObjs{iTT0B+1} = r;
        if obj.treeTblRowIsGroupingRow(r)
          % none
        else
          ctr = ctr+1;
          iD2TT(ctr) = iTT0B+1;
        end
      end
      obj.treeTblRowObjs = ttRowObjs;
      obj.iData2iTTExpanded1B = iD2TT;
      
      mFld2TT0B = containers.Map;
      for i=3:numel(rawHdrs)
        % hdrs{1} is the grouping var; it's not a treeTable col
        % hdrs{2} is a "dummy" var for the 1st treeTable col, col0B==0
        mFld2TT0B(rawHdrs{i}) = i-2;
      end
      obj.fld2ttCol0B = mFld2TT0B;
      
      if ~isempty(colPreferredWidths)
        assert(isa(colPreferredWidths,'containers.Map'));        
        keys = colPreferredWidths.keys;
        vals = cell2mat(colPreferredWidths.values);
        assert(all(ismember(rawHdrs,keys)),...
          'One or more table columns has no column width.');
        if sum(vals)~=1.0
          error('NagivationTreeTable:colwidth',...
            'Column width factors must sum to 1.0.');
        end

        ttWidth = tt.Width;
        for k=keys(:)',k=k{1}; %#ok<FXSET>
          tblCol = tt.getColumn(k);
          wfac = colPreferredWidths(k);          
          tblCol.setPreferredWidth(ttWidth*wfac);
        end
      end
      
      obj.treeTblData = tbl;
      obj.treeTblDataPrettyHeaders = prettyHdrsOrig;
    end
    
    function updateDataRow(obj,iData,fld,val)
      iTTExpanded1B = obj.iData2iTTExpanded1B(iData);
      rowObj = obj.treeTblRowObjs{iTTExpanded1B};
      col0B = obj.fld2ttCol0B(fld);
      rowObj.setValueAt(val,col0B);
    end
    
    function updateDataColumn(obj,fld,valCell)
      % Same as updateDataRow over all rows
      
      nDat = obj.nData;
      if nDat==0
        assert(isempty(valCell));
        return;
      end
      assert(iscell(valCell) && iscolumn(valCell) && numel(valCell)==nDat);
      iData2iTT1B = obj.iData2iTTExpanded1B;
      ttRowObjs = obj.treeTblRowObjs;
      col0B = obj.fld2ttCol0B(fld);
      for iData=1:nDat
        iTTExpanded1B = iData2iTT1B(iData);
        rowObj = ttRowObjs{iTTExpanded1B};
        rowObj.setValueAt(valCell{iData},col0B);
      end
    end
  
    function expandSetI(obj,iSet)
      % Expand specified set, collapse all others. No row is selected.
      %
      % iSet: set index (1-based), where sets are ordered as in data
      
      setIdx0B = iSet-1;
      tt = obj.treeTbl;
      tt.collapseAll;
      tt.expandRow(setIdx0B,true);
    end
    
    function setSelectedDataRow(obj,iData)
      % Select row matching tblrow. tblrow's set is expanded if necessary.
      %
      % iData: 1-based index into rows of data table

      tblData = obj.treeTblData;
      assert(1<=iData && iData<=height(tblData));
      
      % find the "set row" and expand if nec
      tt = obj.treeTbl;
      tblrowSet = tblData{iData,1};
      rowObj = [];
      for i=1:tt.RowCount
        rowObj = tt.getRowAt(i-1);
        [tf,groupval] = obj.treeTblRowIsGroupingRow(rowObj);
        if tf && strcmp(groupval,tblrowSet)
          % found it; rowObj is set, i is the 1-BI for rowObj
          rowObjIdx = i;
          break;
        end
      end
      if isempty(rowObj)
        % couldn't find the set row
        return;
      end
      if ~rowObj.isExpanded
        tt.getModel.expandRow(rowObj,true);
      end
      
      % set the actual row
      % cellstr shouldn't be nec, ML returning char arrays if all sets have
      % same width
      [tffound,ittFirstInSet] = obj.firstDataRowForSet(tblrowSet);
      assert(tffound);
      delRows = iData-ittFirstInSet;
      % delRows in {0,1,...}. 
      ttRow0B = rowObjIdx-1+delRows+1; 
      % first -1, convert to 0-based. last +1, add one to move off of "set
      % row"
      tt.setSelectedRow(tt.getRowAt(ttRow0B)); 
    end

    function iData = getSelectedDataRow(obj)
      % iData: [nsel] rows in data table corresponding to current treeTable
      % selection
      %
      % IMPORTANT: currently CANNOT reorder/sort table 
      
      tt = obj.treeTbl;
      selRow = tt.getSelectedRows;
      iData = arrayfun(@(x)obj.findDataRowFromTreeTableRow(x),selRow);
    end
            
  end
  
  methods (Hidden)
    
    function [tf,groupval] = treeTblRowIsGroupingRow(obj,rowObj)
      rowstr = char(rowObj);
      sRE = regexp(rowstr,obj.groupTreeTblRowREPat,'names');
      tf = ~isempty(sRE);
      if tf
        groupval = sRE.set;
      else
        groupval = [];
      end
    end
    
    function [tffound,iData] = firstDataRowForSet(obj,setval)
      % Find first data row for given set value. 
      %
      % iData: 1-based row into .treeTblData. 
      % tffound: scalar logical
      
      ttData = obj.treeTblData;
      % cellstr shouldn't be nec, ML returning char arrays if all sets have
      % same width
      iData = find(strcmp(cellstr(ttData{:,1}),setval),1);
      tffound = ~isempty(iData);
    end
    
    function iData = findDataRowFromTreeTableRow(obj,iTbl0B)
      % Find the row into the data table (most recent arg to .setData) for
      % 0-based row into treeTable as currently displayed/collapsed etc
      %
      % iTbl0B: 0-based row of treeTable currently selected
      %
      % iData: 1-based row of data table. If iTbl0B represents a "grouping" 
      % treeTable row, the first data row for the given group/set is
      % returned (as if iTbl0B was incremented by on onto the next
      % "content" row).
      
      % find the set/group row in table for iTbl0B
      tt = obj.treeTbl;
      ittDel = nan;
      for itt = iTbl0B:-1:0
        rowObj = tt.getRowAt(itt);
        [tf,groupval] = obj.treeTblRowIsGroupingRow(rowObj);
        if tf
          if itt==iTbl0B
            ittDel = 1; % as if iTbl0B were one larger
          else
            ittDel = iTbl0B-itt;
          end
          break;
        end
      end
      
      assert(~isnan(ittDel)); % All rows have a parent/group row
      
      % find first data row for given group, then increment by ittDel
      [tffound,iData] = obj.firstDataRowForSet(groupval);
      assert(tffound);
      iData = iData + ittDel - 1; % -1 b/c when ittDel==1, you don't want any increment
    end
    
    function navSelected(obj)
      iData = obj.getSelectedDataRow();
      if isempty(iData)
        % none
      else
        if numel(iData)>1
          warningNoTrace('NavigationTreeTable:rows',...
            'Multiple rows selected. Using first row selected.');
        end
        obj.fcnDataRowNaved(iData(1));
      end
    end
    
    function cbkTableClick(obj,src,evt)
      persistent chk
      PAUSE_DURATION_CHECK = 0.25;
      
      if obj.navOnSingleClick
        obj.navSelected();
        return;
      end
      
      if isempty(chk)
        chk = 1;
        pause(PAUSE_DURATION_CHECK); %Add a delay to distinguish single click from a double click
        if chk==1
          % single-click
          chk = [];
        end
      else
        % double-click
        chk = [];
        obj.navSelected();
      end
    end
    
  end
  
  methods (Static)
    
    function verifyDataTable(dataTbl)
      % NavigationTreeTable assumes that the data table comes with rows
      % sorted/bunched by the grouping variable, ie all group1's come
      % first, then all group2's, etc.
      
      g = dataTbl{:,1};
      if ~iscellstr(g)
        error('NavigationTreeTable:dataTable',...
          'Grouping variable must be cellstr.');
      end
      [~,~,idx] = unique(g,'stable');
      if all(diff(idx)>=0)
        % g monotonically increasing;
      else        
        error('NavigationTreeTable:dataTable',...
          'Unexpected data table grouping variable values.');
      end
    end
    
    function ty = col2type(var)
      if iscellstr(var)
        ty = 'char';
      else
        matvar = cell2mat(var); % should prob try/catch
        if islogical(matvar)
          ty = 'logical';
        elseif isnumeric(matvar) 
          if all(matvar==round(matvar))
            ty = 'int';
          else
            ty = 'double';
          end
        else
          assert(false);
        end
      end
    end
    
  end
  
end