classdef TrainingOverlayReceiver < handle
  properties
    axs
    tstrBases
    tblMFT
  end
  methods
    function obj = TrainingOverlayReceiver(hAx,tbases,tMFT)
      assert(numel(hAx)==numel(tbases));
      obj.axs = hAx; % can be nonscalar
      obj.tstrBases = tbases;
      obj.tblMFT = tMFT;
    end      
    function respond(obj,eid)
      if isempty(eid)
        for i=1:numel(obj.axs)
          title(obj.axs(i),obj.tstrBases{i},'fontweight','bold');
        end        
      else
        trow = obj.tblMFT(eid,:);
        for i=1:numel(obj.axs)          
          tstr = sprintf('%s \\color{darkgreen}Selection: (mov %d,frm %d,tgt %d)',...
            obj.tstrBases{i},int32(trow.mov),trow.frm,trow.iTgt);
          title(obj.axs(i),tstr,'fontweight','bold');
        end
      end
    end
  end
end
  