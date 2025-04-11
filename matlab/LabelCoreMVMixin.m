classdef LabelCoreMVMixin < handle
% Multiview labeling mixin class
  
  properties (Abstract) 
    hAx
    nView
    nPointSet
    txLblCoreAux
    labeler
  end
  
  properties (SetAccess=private)
    numHotKeySet; % row index into numHotKeySetInfo for currently active hot key set
    numHotKeySetInfo; % 6 columns; see genNumHotKeySetInfo()
    
    hAxXLabels; % [nview] xlabels for axes
    hAxXLabelsFontSize = 11;
  end
  
  methods
    
    function delete(obj)
      deleteValidGraphicsHandles(obj.hAxXLabels);
      obj.hAxXLabels = [];
    end
    
    function initHook(obj)
      obj.hAxXLabels = gobjects(obj.nView,1);
      for iView=1:obj.nView
        ax = obj.hAx(iView);
        obj.hAxXLabels(iView) = xlabel(ax,'','fontsize',obj.hAxXLabelsFontSize);
      end
      
      obj.numHotKeySetInfo = LabelCoreMVMixin.genNumHotKeySetInfo(...
        obj.nView,obj.nPointSet);
      obj.numHotKeySet = 1;
      obj.refreshHotkeyDesc();
    end
    
  end
  
  %% Num HotKeys
  methods
    
    function incrementHotKeySet(obj)
      obj.numHotKeySet = obj.numHotKeySet+1;
      if obj.numHotKeySet>size(obj.numHotKeySetInfo,1)
        obj.numHotKeySet = 1;
      end
      obj.refreshHotkeyDesc();
    end
    
    function decrementHotKeySet(obj)
      obj.numHotKeySet = obj.numHotKeySet-1;
      if obj.numHotKeySet<1
        obj.numHotKeySet = size(obj.numHotKeySetInfo,1);
      end
      obj.refreshHotkeyDesc();
    end
    
    function [tf,iPt] = numHotKeyInRange(obj,keyval)
      % keyval: 1-10
      %
      % tf: true if keyval is in range
      % iPt: iPt corresponding to keyval
      hkset = obj.numHotKeySetInfo(obj.numHotKeySet,:);
      loKey = hkset(4);
      hiKey = hkset(5);
      loPt = hkset(6);
      
      tf = loKey<=keyval && keyval<=hiKey;
      if tf
        iPt = loPt + keyval - loKey;
      else
        iPt = nan;
      end
    end
    
    function refreshHotkeyDesc(obj)
      hkset = obj.numHotKeySetInfo(obj.numHotKeySet,:);
      iView = hkset(1);
      loPtSet = hkset(2);
      hiPtSet = hkset(3);
      loKey = hkset(4);
      hiKey = hkset(5);
      
      viewName = obj.labeler.viewNames{iView};
      str = sprintf('Hotkeys ''%d''->''%d'': view:%s pts%d->%d',...
        loKey,hiKey,viewName,loPtSet,hiPtSet);
      [obj.hAxXLabels(2:end).String] = deal(str);
      obj.txLblCoreAux.String = str;
    end
    
  end
  methods (Static)
    
    function info = genNumHotKeySetInfo(nView,nPointSet)
      % info: 7 cols.
      %  1. iView
      %  2. lo pointset in view
      %  3. hi pointset in view
      %  4. lo key (1 through 0==10)
      %  5. hi key (")
      %  6. lo iPt
      %  7. hi iPt
      %
      % Each row of info gives you a current state for num hotkeys. Let's
      % say row = info(i,:). Then hitting hotkeys row(4) through row(5)
      % maps to 3dpoint row(2) through row(3) in view row(1).
      nGroupsOf10PerView = ceil(nPointSet/10);
      info = nan(0,7);
      for iView = 1:nView
        tmp = [repmat(iView,nGroupsOf10PerView,1) (1:10:nPointSet)'];
        tmp(:,3) = min(tmp(:,2)+9,nPointSet);
        tmp(:,4) = 1;
        tmp(:,5) = tmp(:,4)+(tmp(:,3)-tmp(:,2));
        tmp(:,6) = (tmp(:,1)-1)*nPointSet+tmp(:,2);
        tmp(:,7) = (tmp(:,1)-1)*nPointSet+tmp(:,3);
        info = cat(1,info,tmp);
      end
    end
    
  end
  
end
