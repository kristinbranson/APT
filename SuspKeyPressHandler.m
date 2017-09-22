classdef SuspKeyPressHandler < handle
  % Ctrl-right/left moves to next/prev selected row in suspSelectedMFT
  % navTable. If no row is selected, nothing is done.
  
  properties
    hNavTable; % suspSelectedMFT NavigationTable
  end
  
  methods
    
    function obj = SuspKeyPressHandler(navTbl)
      assert(isa(navTbl,'NavigationTable'));
      obj.hNavTable = navTbl;
    end
    
    function tfHandled = handleKeyPress(obj,evt)
      tfHandled = false;
      if any(strcmp('control',evt.Modifier)) && numel(evt.Modifier)==1
        switch evt.Key
          case 'rightarrow'
            drow = 1;
            tfTryHandling = true;
          case 'leftarrow'
            drow = -1;
            tfTryHandling = true;
          otherwise
            tfTryHandling = false;
        end
        if tfTryHandling
          nt = obj.hNavTable;
          rows = nt.getSelectedRows;
          if ~isempty(rows)
            tfHandled = true;
            newrow = rows(1)+drow;
            if 1<=newrow && newrow<=nt.height
              nt.setSelectedRows(newrow);
            end
          end
        end
      end
    end
    
  end
  
end