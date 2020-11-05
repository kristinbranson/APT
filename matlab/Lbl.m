classdef Lbl
  methods (Static)
    function writeTrnPack(lObj,packdir,varargin)
      writeims = myparse(varargin,...
        'writeims',false ...
        );
      
      sagg = Lbl.aggregateLabelsAddRoi(lObj);
      sloc = Lbl.writeIms(sagg,lObj.movieInfoAll,packdir,'writeims',writeims);
      
      % base/raw package
      j = jsonencode(sagg);
      jsonoutf = 'trnpack.json';
      jsonoutf = fullfile(packdir,jsonoutf);
      fh = fopen(jsonoutf,'w');
      fprintf(fh,'%s\n',j);
      fclose(fh);
      fprintf(1,'Wrote %s.\n',jsonoutf);
      
      %% reduced pack
      slocjse = jsonencode(sloc);
      slocjsf = 'loc.json';
      slocjsf = fullfile(packdir,slocjsf);
      fh = fopen(slocjsf,'w');
      fprintf(fh,'%s\n',slocjse);
      fclose(fh);
      fprintf(1,'Wrote %s\n',slocjsf);
    end
    function sagg = aggregateLabelsAddRoi(lObj)
      nmov = numel(lObj.labels);
      sagg = cell(nmov,1);
      for imov=1:nmov
        s = lObj.labels{imov};
        s.mov = lObj.movieFilesAllFull{imov};
        
        %% gen rois, bw
        n = size(s.p,2);
        s.roi = nan(8,n);
        fprintf(1,'mov %d: %d labeled frms.\n',imov,n);
        for i=1:n
          p = s.p(:,i);
          xy = Shape.vec2xy(p);
          roi = lObj.maGetRoi(xy);
          s.roi(:,i) = roi(:);
        end
        
        sagg{imov} = s;
      end
      sagg = cell2mat(sagg);
    end
    function sloc = writeIms(sagg,movInfoAll,packdir,varargin)
      % write trnpack images, return loc info
      
      writeims = myparse(varargin,...
        'writeims',false);
      
      assert(numel(sagg)==numel(movInfoAll));
      mr = MovieReader;
            
      if writeims
        SUBDIRMASK = 'mask';
        SUBDIRIM = 'im';
        SUBDIRIMMASK = 'immask';
        SUBDIRIMMASKC = 'immaskc';
        for sdir={SUBDIRMASK SUBDIRIM SUBDIRIMMASK SUBDIRIMMASKC},sdir=sdir{1}; %#ok<FXSET>
          if exist(fullfile(packdir,sdir),'dir')==0
            mkdir(packdir,sdir);
          end
        end
      end
      
      sloc = [];
      for imov=1:numel(sagg)
        s = sagg(imov);
        
        mr.open(s.mov);
        
        movifo = movInfoAll{imov};
        imsz = [movifo.info.nr movifo.info.nc];
        fprintf(1,'mov %d (sz=%s): %s\n',imov,mat2str(imsz),s.mov);
        
        frmsun = unique(s.frm);
        nfrmsun = numel(frmsun);
        %maskfullsz = [IMSZ nfrmsun];
        %s.maskfull = nan(maskfullsz);
        for ifrmun=1:nfrmsun
          f = frmsun(ifrmun);
          idx = find(s.frm==f);
          ntgt = numel(idx);
          itgt = s.tgt(idx);
          mask = false(imsz);
          for j=idx(:)'
            bw = poly2mask(s.roi(1:4,j),s.roi(5:8,j),imsz(1),imsz(2));
            mask = mask | bw;
          end
          xyf = reshape(s.p(:,idx),s.npts,2,ntgt); % shapes for all tgts in this frm
          ts = reshape(s.ts(:,idx),s.npts,ntgt); % ts "
          occ = reshape(s.occ(:,idx),s.npts,ntgt); % estocc "
          
          maskc = any(mask,1);
          maskr = any(mask,2);
          c0 = find(maskc,1,'first');
          c1 = find(maskc,1,'last');
          r0 = find(maskr,1,'first');
          r1 = find(maskr,1,'last');
          
          roicrop = [c0 c1 r0 r1];
          xyfcrop = xyf;
          xyfcrop(:,1,:) = xyfcrop(:,1,:)-c0+1;
          xyfcrop(:,2,:) = xyfcrop(:,2,:)-r0+1;
          
          imfrm = mr.readframe(f);
          imfrmmask = imfrm;
          imfrmmask(~mask) = 0;
          imfrmmaskcrop = imfrmmask(r0:r1,c0:c1);
          
          basefS = sprintf('mov%04d_frm%08d_tgt%03d',imov,f,ntgt);
          if writeims
            basefSpng = [basefS '.png'];            
            maskf = fullfile(packdir,SUBDIRMASK,basefSpng);
            imfrmf = fullfile(packdir,SUBDIRIM,basefSpng);
            imfrmmaskf = fullfile(packdir,SUBDIRIMMASK,basefSpng);
            imfrmmaskcropf = fullfile(packdir,SUBDIRIMMASKC,basefSpng);
            
            imwrite(mask,maskf);
            imwrite(imfrm,imfrmf);
            imwrite(imfrmmask,imfrmmaskf);
            imwrite(imfrmmaskcrop,imfrmmaskcropf);
            fprintf(1,'Wrote files for %s...\n',basefS);
          else
            fprintf(1,'Didnt write files for %s...\n',basefS);
          end
          sloctmp = struct(...
            'id',basefS,...
            'imov',imov,...
            'mov',s.mov,...
            'frm',f,...
            'ntgt',ntgt,...
            'itgt',itgt,...
            'roiccrop',roicrop, ...
            'xyabs',xyf, ...
            'xycrop',xyfcrop, ...
            'occ',occ, ...
            'ts',ts ...
            );
          sloc = [sloc; sloctmp]; %#ok<AGROW>
        end
      end
    end    
    function s = createStrippedLblsUseTopLevelTrackParams(lObj,iTrkers,...
        varargin)
      % Create/save a series of stripped lbls based on current Labeler proj
      %
      % lObj: Labeler obj with proj loaded
      % iTrkers: vector of tracker indices for which stripped lbl will be 
      %   saved
      %
      % s: cell array of stripped lbls
      %
      % This method exists bc:
      % - Strictly speaking, stripped lbls are net-specific, as setting
      % base tracking parameters onto a DeepTracker obj has hooks/codepath
      % for mutating params.
      % - Sometimes, you want to generate a stripped lbl from the top-level
      % params which are not yet set on a particular tracker.
      %
      % This method is here rather than Labeler bc Labeler is getting big.
            
      [docompress,dosave] = myparse(varargin,...
        'docompress',true, ...
        'dosave',true ... save stripped lbls (loc printed)
        );
      
      ndt = numel(iTrkers);
      s = cell(ndt,1);
      for idt=1:ndt
        itrker = iTrkers(idt);
        lObj.trackSetCurrentTracker(itrker);
        tObj = lObj.tracker;
  
        tObj.setAllParams(lObj.trackGetParams()); % does not set skel, flipLMEdges
        sthis = tObj.trnCreateStrippedLbl();
        if docompress
          sthis = Lbl.compressStrippedLbl(sthis);
        end
        
        s{idt} = sthis;
        
        if dosave
          fsinfo = lObj.projFSInfo;
          [lblP,lblS] = myfileparts(fsinfo.filename);
          sfname = sprintf('%s_%s.lbl',lblS,tObj.algorithmName);
          sfname = fullfile(lblP,sfname);
          save(sfname,'-mat','-struct','sthis');
          fprintf(1,'Saved %s\n',sfname);
        end
      end
      
    end
    function s = compressStrippedLbl(s)
      CFG_GLOBS = {'Num'};
      FLDS = {'cfg' 'projname' 'projMacros' 'movieInfoAll' 'cropProjHasCrops' ...
        'trackerClass' 'trackerData'};
      GLOBS = {'labeledpos' 'movieFilesAll' 'trxFilesAll' 'preProcData'};
      
      fldscfg = fieldnames(s.cfg);      
      fldscfgkeep = fldscfg(startsWith(fldscfg,CFG_GLOBS));
      s.cfg = structrestrictflds(s.cfg,fldscfgkeep);
      
      flds = fieldnames(s);
      fldskeep = flds(startsWith(flds,GLOBS));
      fldskeep = [fldskeep(:); FLDS(:)];
      s = structrestrictflds(s,fldskeep);
    end
  end
end