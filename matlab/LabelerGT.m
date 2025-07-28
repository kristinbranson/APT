classdef LabelerGT
  % GT-related utils for Labeler. Labeler.m is getting very large; nav is
  % getting difficulty, GH blame breaks etc. This 'friend class' has only
  % access to public props atm but in future could theoretically use access
  % specifiers etc.
  
  methods (Static)
    
    function generateSuggestionsUI(lObj)
      % Use GTSuggest UI to auto-generate GT frames
      
      if ~isempty(lObj.gtSuggMFTable) && any(lObj.gtSuggMFTableLbled)
        qmsg = 'Frames to label for groundtruthing have already been selected and some have been labeled. Are you sure you want to re-choose the list of frames to label?';
        resp = questdlg(qmsg,'Groundtruthing already started','OK, proceed','Cancel','OK, proceed');
        if isempty(resp)
          resp = 'Cancel';
        end
        switch resp
          case 'OK, proceed'
            % none
          case 'Cancel'
            return;
          otherwise
            assert(false);
        end
      end
      
      % Note, any existing labels are not deleted. On gtCompute these other
      % labels are not currently used however.
      
      gtsg = GTSuggest(lObj);
      if ~isempty(gtsg)
        tblGT = gtsg.generate(lObj);
        lObj.gtSetUserSuggestions(tblGT,'sortcanonical',true);
      else
        % user canceled or similar
      end
    end
    
    function saveSuggestionsUI(lObj)
      tbl = lObj.gtSuggMFTable;
      if isempty(tbl),
        warndlg('To-label list is empty','To-label list is empty','modal');
        return;
      end
      gtsuggmat = lObj.rcGetProp('gtsuggestionsmat');
      if isempty(gtsuggmat)
        gtsuggmat = pwd;
      end
      [fname,pth] = uiputfile('*.mat','Save to-label list',gtsuggmat);
      if isequal(fname,0)
        return;
      end
      fname = fullfile(pth,fname);
      lObj.rcSaveProp('gtsuggestionsmat',fname);
      save(fname,'tbl','-mat');
    end

    function loadSuggestionsUI(lObj)
      gtsuggmat = lObj.rcGetProp('gtsuggestionsmat');
      if isempty(gtsuggmat)
        gtsuggmat = pwd;
      end
      [fname,pth] = uigetfile('*.mat','Load to-label list from MAT file',gtsuggmat);
      if isequal(fname,0)
        return;
      end
      fname = fullfile(pth,fname);
      lObj.rcSaveProp('gtsuggestionsmat',fname);
      
      assert(lObj.gtIsGTMode);
      tbl = MFTable.loadTableFromMatfile(fname);
      if ~isnumeric(tbl.mov)
        [tffound,mIdx] = lObj.getMovIdxMovieFilesAllFull(tbl.mov,'gt',true);
        if any(~tffound)
          errstrs = {'Moviesets in table not found in project:'};
          movstrsnotfound = MFTable.formMultiMovieIDArray(tbl.mov(~tffound,:),...
            'separator',',','checkseparator',false);
          errstrs = [errstrs; movstrsnotfound];
          errordlg(errstrs,'Moviesets not found');
          return;
        end
        
        szassert(mIdx,[height(tbl) 1]);
        assert(isa(mIdx,'MovieIndex'));
        [~,gt] = mIdx.get();
        assert(all(gt));
        tbl.mov = mIdx;
      end
      
      lObj.gtSetUserSuggestions(tbl);
      msgstr = sprintf('Loaded to-label list for groundtruthing with %d examples spanning %d GT movies.',...
        height(tbl),numel(unique(tbl.mov)));
      msgbox(msgstr,'GT Table Loaded');
    end
    
    function setSuggestionsToLabeledUI(lObj)
      assert(lObj.gtIsGTMode);
      lObj.gtSetUserSuggestions([]);
      tbl = lObj.gtSuggMFTable;
      msgstr = sprintf('Set to-label list for groundtruthing to %d examples spanning %d GT movies.',...
        height(tbl),numel(unique(tbl.mov)));
      msgbox(msgstr,'GT Table Loaded');
    end
    
  end
  
end