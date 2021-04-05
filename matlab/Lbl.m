classdef Lbl
  
  methods (Static) % ma train package
    
    function vizLoc(packdir,varargin)
      % Visualize 'loc' data structure (one row per labeled mov,frm,tgt) 
      % from training package.
      %
      % packdir: package dir (contains images)

      [scargs,ttlargs] = myparse(varargin,...
        'scargs',{16}, ...
        'ttlargs',{'fontsize',16,'fontweight','bold','interpreter','none'} ...
        );
      
      [~,~,loc,~] = Lbl.loadPack(packdir);
       
      hfig = figure(11);
      
      idmfun = unique({loc.idmovfrm}');
      nmfun = numel(idmfun);
      for iidmf=1:nmfun
        idmf = idmfun{iidmf};
        is = find(strcmp({loc.idmovfrm}',idmf));
        
        imf = fullfile(packdir,'im',[idmf '.png']);
        if exist(imf,'file')==0
          warningNoTrace('Skipping, image not found: %s',imf);
          continue;
        end
          
        im = imread(imf);
        
        clf;
        ax = axes;
        imagesc(im);
        colormap gray;
        hold on;
        axis square;
        
        for j = is(:)'
          s = loc(j);
          xy = reshape(s.p,[],2);        
          scatter(xy(:,1),xy(:,2),scargs{:});        
          plot(s.roi([1:4 1]),s.roi([5:8 5]),'r-','linewidth',2);
        end
        
        tstr = sprintf('%s: %d tgts',idmf,numel(is));
        title(tstr,ttlargs{:});
        input(idmf);
      end        
    end
    
    function vizLocg(packdir,varargin)
      % Visualize 'locg' data structure (one row per labeled mov,frm) 
      % from training package.
      %
      % packdir: package dir (contains images)

      [scargs,ttlargs,frms,locg] = myparse(varargin,...
        'scargs',{16}, ...
        'ttlargs',{'fontsize',16,'fontweight','bold','interpreter','none'}, ...
        'frms', [], ... % opt; frames (indices into locg.locdata) to viz
        'locg', [] ... %  opt; preloaded locg/json
        );
      
      if isempty(locg)
        [~,~,~,locg] = Lbl.loadPack(packdir);
      end

      hfig = figure(11);
      
      if isempty(frms)
        nfrm = numel(locg.locdata);      
        frms = 1:nfrm;
      end
      for ifrm=frms(:)'
        if iscell(locg.locdata)
          s = locg.locdata{ifrm};
        else
          s = locg.locdata(ifrm);
        end
        imf = fullfile(packdir,s.img);
        if iscell(imf)
          assert(isscalar(imf));
          imf = imf{1};
        end
        im = imread(imf);
        
        clf;
        ax = axes;
        imagesc(im);
        colormap gray;
        hold on;
        axis square;
        
        for itgt=1:s.ntgt
          if isfield(s,'pabs')
            xy = reshape(s.pabs(:,itgt),[],2);
            scatter(xy(:,1),xy(:,2),scargs{:});
          end
          plot(s.roi([1:4 1],itgt),s.roi([5:8 5],itgt),'r-','linewidth',2);
        end
        
        if isfield(s,'extra_roi')
          nroi = size(s.extra_roi,2);
          for j=1:nroi
            plot(s.extra_roi([1:4 1],j),s.extra_roi([5:8 5],j),'b-','linewidth',2);
          end
        end        
        
        tstr = sprintf('%s: %d tgts',s.id,s.ntgt);
        title(tstr,ttlargs{:});
        input(tstr);
      end        
    end
    
    function vizLocClus(packdir,varargin)
      % Visualize 'locg' data structure (one row per labeled mov,frm)
      % from training package.
      %
      % packdir: package dir (contains images)
      
      [scargs,ttlargs] = myparse(varargin,...
        'scargs',{16}, ...
        'ttlargs',{'fontsize',16,'fontweight','bold','interpreter','none'} ...
        );
      
      [~,~,~,~,loccc] = Lbl.loadPack(packdir);

      hfig = figure(11);
      
      ncluster = numel(loccc);
      for iclus=1:ncluster
        s = loccc(iclus);
        assert(isscalar(s.img));
        imf = fullfile(packdir,s.img{1});
        im = imread(imf);
        
        clf;
        ax = axes;
        imagesc(im);
        colormap gray;
        hold on;
        axis square;
        
        for itgt=1:s.ntgt
          xy = reshape(s.pabs(:,itgt),[],2);
          scatter(xy(:,1),xy(:,2),scargs{:});
        end
        szassert(s.roi,[8 1]);
        plot(s.roi([1:4 1]),s.roi([5:8 5]),'r-','linewidth',2);
        
        tstr = sprintf('%s: %d tgts',s.idclus,s.ntgt);
        title(tstr,ttlargs{:});
        input(tstr);
      end
    end
      
    function s = hlpLoadJson(jsonfile)
      jse = readtxtfile(jsonfile);
      s = jsondecode(jse{1});
      fprintf(1,'loaded %s\n',jsonfile);
    end
    function [slbl,tp,loc,locg,loccc] = loadPack(packdir)
      % Load training package into MATLAB data structures
      %
      % slbl: 'stripped lbl' struct
      % tp: one-row-per-movie struct. Maybe a useful format for metadata 
      %   or bookkeeping purposes.
      % loc: one-row-per-labeled-(mov,frm,tgt) struct. Intended to be
      %   primary MA keypt data structure.
      % loccc: one-row-per-labeled-cluster. Experimental, may not be
      %   useful for DL/py backend.
      %
      % Note tp, loc, loccc contain equivalent info just in different
      % formats.
       
      dd = dir(fullfile(packdir,'*.lbl'));
      if ~isscalar(dd)
        lbln = {dd.name}';
        lbln = sort(lbln);
        warningNoTrace('%d .lbl files found. Using: %s',numel(lbln),lbln{end});
        lblsf = lbln{end};
      else
        lblsf = dd.name;
        fprintf(1,'Using lbl: %s\n',lblsf);
      end
        
      lblsf = fullfile(packdir,lblsf);
      slbl = load(lblsf,'-mat');
      fprintf(1,'loaded %s\n',lblsf);
      
      tpf = fullfile(packdir,'trnpack.json');
      tp = Lbl.hlpLoadJson(tpf);

      locf = fullfile(packdir,'loc0.json');
      loc = Lbl.hlpLoadJson(locf);

      locf = fullfile(packdir,'loc.json');
      locg = Lbl.hlpLoadJson(locf);

      locf = fullfile(packdir,'locclus.json');
      if exist(locf,'file')>0
        locjse = readtxtfile(locf);
        loccc = jsondecode(locjse{1});
        fprintf(1,'loaded %s\n',locf);
      else
        loccc = [];
      end
    end
    
    function hlpSaveJson(s,packdir,jsonoutf)
      j = jsonencode(s);
      jsonoutf = fullfile(packdir,jsonoutf);
      fh = fopen(jsonoutf,'w');
      fprintf(fh,'%s\n',j);
      fclose(fh);
      fprintf(1,'Wrote %s.\n',jsonoutf);
    end
    function [slbl,tp,loc,locg,loccc] = genWriteTrnPack(lObj,packdir,varargin)
      % Generate training package. Write contents (raw images and keypt 
      % jsons) to packdir.
      
      [writeims,writeimsidx,slblname] = myparse(varargin,...
        'writeims',true, ...
        'writeimsidx',[], ...
        'strippedlblname',[] ... % (opt) short filename for stripped lbl
        );
      
      if exist(packdir,'dir')==0
        mkdir(packdir);
      end
      
      tObj = lObj.tracker;
      tObj.setAllParams(lObj.trackGetParams()); % does not set skel, flipLMEdges
      slbl = tObj.trnCreateStrippedLbl();
      slbl = Lbl.compressStrippedLbl(slbl,'ma',true);
      [~,jslbl] = Lbl.jsonifyStrippedLbl(slbl);
      
      fsinfo = lObj.projFSInfo;
      [lblP,lblS] = myfileparts(fsinfo.filename);
      if isempty(slblname)
        slblname = sprintf('%s_%s.lbl',lblS,tObj.algorithmName);
      end
      sfname = fullfile(packdir,slblname);
      save(sfname,'-mat','-v7.3','-struct','slbl');
      fprintf(1,'Saved %s\n',sfname);
      [~,slblnameS] = fileparts(slblname);
      sfjname = sprintf('%s.json',slblnameS);
      Lbl.hlpSaveJson(jslbl,packdir,sfjname);
     

      tp = Lbl.aggregateLabelsAddRoi(lObj);
      [loc,locg,loccc] = Lbl.genLocs(tp,lObj.movieInfoAll);
      if writeims
        if isempty(writeimsidx)
          writeimsidx = 1:numel(loc);
        end
        Lbl.writeims(loc(writeimsidx),packdir);
      end
        
      % trnpack: one row per mov
      jsonoutf = 'trnpack.json';
      Lbl.hlpSaveJson(tp,packdir,jsonoutf);
      
      % loc: one row per labeled tgt
      jsonoutf = 'loc0.json';
      Lbl.hlpSaveJson(loc,packdir,jsonoutf);

      % loc: one row per frm
      jsonoutf = 'loc.json';
      s = struct();
      s.movies = lObj.movieFilesAllFull;
      s.splitnames = {'trn'};
      s.locdata = locg;
      Lbl.hlpSaveJson(s,packdir,jsonoutf);

      % loccc: one row per cluster
      jsonoutf = 'locclus.json';
      Lbl.hlpSaveJson(loccc,packdir,jsonoutf);      
    end
    
    function sagg = aggregateLabelsAddRoi(lObj)
      nmov = numel(lObj.labels);
      sagg = cell(nmov,1);
%       saggroi = lObj.labelsRoi;
%       szassert(saggroi,size(sagg));
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

        sroi = lObj.labelsRoi{imov};
        s.frmroi = sroi.f;
        s.extra_roi = sroi.verts;
        sagg{imov} = s;        
      end
      sagg = cell2mat(sagg);
    end
    function [sloc,slocg,sloccc] = genLocs(sagg,movInfoAll)
      assert(numel(sagg)==numel(movInfoAll));
      nmov = numel(sagg);
      sloc = [];
      slocg = [];
      sloccc = [];
      for imov=1:nmov
        s = sagg(imov);
        movifo = movInfoAll{imov};
        imsz = [movifo.info.nr movifo.info.nc];
        fprintf(1,'mov %d (sz=%s): %s\n',imov,mat2str(imsz),s.mov);
        
        slocI = Lbl.genLocsI(s,imov);
        slocgI = Lbl.genLocsGroupedI(s,imov);
        slocccI = Lbl.genCropClusteredLocsI(s,imsz,imov);
        
        sloc = [sloc; slocI]; %#ok<AGROW>
        slocg = [slocg; slocgI]; %#ok<AGROW>
        sloccc = [sloccc; slocccI]; %#ok<AGROW>
      end
    end
    function [sloc] = genLocsI(s,imov,varargin)      
      imgpat = myparse(varargin,...
        'imgpat','im/%s.png' ...
        );
      
      sloc = [];
      nrows = size(s.p,2);
      for j=1:nrows        
        f = s.frm(j);
        itgt = s.tgt(j);
        ts = s.ts(:,j);
        occ = s.occ(:,j);
        roi = s.roi(:,j);
        
        basefS = sprintf('mov%04d_frm%08d',imov,f);
        img = sprintf(imgpat,basefS);
        sloctmp = struct(...
          'id',sprintf('mov%04d_frm%08d_tgt%03d',imov,f,itgt),...
          'idmovfrm',sprintf('mov%04d_frm%08d',imov,f),...
          'img',{{img}},...
          'imov',imov,...
          'mov',s.mov,...
          'frm',f,...
          'itgt',itgt,...
          'roi',roi,...
          'p',s.p(:,j), ...
          'occ',occ, ...
          'ts',ts ...
          );
        sloc = [sloc; sloctmp]; %#ok<AGROW>
      end
    end
    function [slocgrp] = genLocsGroupedI(s,imov,varargin)
      % s: scalar element of 'sagg', ie labels data structure for one movie.
      % imov: movie index, only used for metadata
      
      imgpat = myparse(varargin,...
        'imgpat','im/%s.png' ...
        );
      
      s = Labels.addsplitsifnec(s);

      slocgrp = [];
      frmsun = unique([s.frm(:); s.frmroi(:)]);
      nfrmsun = numel(frmsun);
      for ifrmun=1:nfrmsun
        f = frmsun(ifrmun);
        j = find(s.frm==f);
        ntgt = numel(j);
        
        jroi = find(s.frmroi==f);
        nroi = numel(jroi);
                    
        % Dont include numtgts, eg what if a target is added to an
        % existing frame.
        basefS = sprintf('mov%04d_frm%08d',imov,f);
        img = sprintf(imgpat,basefS);
        sloctmp = struct(...
          'id',basefS,...
          'img',{{img}},...
          'imov',imov,... % 'mov',s.mov,...
          'frm',f,...
          'ntgt',ntgt,...
          'split',s.split(j),...
          'itgt',s.tgt(j),...
          'roi',s.roi(:,j),...
          'pabs',s.p(:,j), ...
          'occ',s.occ(:,j), ...
          'ts',s.ts(:,j), ...
          'nextra_roi',nroi,...
          'extra_roi',reshape(s.extra_roi(:,:,jroi),[],nroi) ...
          );
        slocgrp = [slocgrp; sloctmp]; %#ok<AGROW>
      end
    end
    function [sloccc] = genCropClusteredLocsI(s,imsz,imov,varargin)
      % s: scalar element of 'sagg', ie labels data structure for one movie.
      % imsz: [nr nc]
      % imov: movie index, only used for metadata
      
      imgpat = myparse(varargin,...
        'imgpat','im/%s.png' ...
        );
      
      sloccc = [];
      frmsun = unique(s.frm);
      nfrmsun = numel(frmsun);
      for ifrmun=1:nfrmsun
        f = frmsun(ifrmun);
        idx = find(s.frm==f);
        ntgt = numel(idx);
        %itgt = s.tgt(idx);
        masks = false([imsz ntgt]);
        for itgt=1:ntgt
          j = idx(itgt);
          bw = poly2mask(s.roi(1:4,j),s.roi(5:8,j),imsz(1),imsz(2));
          masks(:,:,itgt) = bw;
        end
        masksAll = any(masks,3);
        
        cc = bwconncomp(masksAll);
        ncc = cc.NumObjects;
        % set of tgts/js in each cc
        js = cell(ncc,1);
        for icc=1:ncc
          maskcc = false(imsz);
          maskcc(cc.PixelIdxList{icc}) = true;
          maskcc = repmat(maskcc,1,1,ntgt);
          maskcc = maskcc & masks; % masks, but restricted to this cc
          tftgtslive = any(any(maskcc,1),2);
          szassert(tftgtslive,[1,1,ntgt]);
          js{icc} = idx(tftgtslive(:));
        end          
        jsall = cat(1,js{:});
        % Each tgt/j should appear in precisely one cc        
        assert(isequal(sort(jsall),sort(idx)));
        
        for icc=1:ncc
          jcc = js{icc};
          ntgtcc = numel(jcc);
          itgtcc = s.tgt(jcc);
          xyf = reshape(s.p(:,jcc),s.npts,2,ntgtcc); % shapes for all tgts in this cc
          ts = reshape(s.ts(:,jcc),s.npts,ntgtcc); % ts "
          occ = reshape(s.occ(:,jcc),s.npts,ntgtcc); % estocc "
          
          [rcc,ccc] = ind2sub(size(masks),cc.PixelIdxList{icc});
          c0 = min(ccc);
          c1 = max(ccc);
          r0 = min(rcc);
          r1 = max(rcc);
          
          roicrop = [c0 c1 r0 r1];
          roi = [c0 c0 c1 c1 r0 r1 r1 r0]';
          xyfcrop = xyf;
          xyfcrop(:,1,:) = xyfcrop(:,1,:)-c0+1;
          xyfcrop(:,2,:) = xyfcrop(:,2,:)-r0+1;                    
          
          % Dont include numtgts, eg what if a target is added to an
          % existing frame.
          basefSclus = sprintf('mov%04d_frm%08d_cc%03d',imov,f,icc);
          %basefSimfrm = sprintf('mov%04d_frm%08d',imov,f);
          
          % one row per CC
          basefS = sprintf('mov%04d_frm%08d',imov,f);
          img = sprintf(imgpat,basefS);       
          sloctmp = struct(...
            'id',basefS,...
            'idclus',basefSclus,...
            'img',{{img}},...
            'imov',imov,...
            'mov',s.mov,...
            'frm',f,...
            'cc',icc,...
            'ntgt',ntgtcc,...
            'itgt',itgtcc,...
            'roi',roi,...% 'roicrop',roicrop, ...
            'pabs',reshape(xyf,[2*s.npts ntgtcc]), ...
            'pcrop',reshape(xyfcrop,[2*s.npts ntgtcc]), ...
            'occ',occ, ...
            'ts',ts ...
            );
          sloccc = [sloccc; sloctmp]; %#ok<AGROW>
        end
      end
    end
    
    function writeims(sloc,packdir)
      % Currently single-view only
      
      SUBDIRIM = 'im';
      sdir = SUBDIRIM;
      if exist(fullfile(packdir,sdir),'dir')==0
        mkdir(packdir,sdir);
      end
      
      imovall = [sloc.imov]';
      imovun = unique(imovall);
      %mr = MovieReader;
      for imov=imovun(:)'
        idx = find(imovall==imov); % indices into sloc for this mov
        % idx cannot be empty
        mov = sloc(idx(1)).mov;
        %mr.open(mov);
        fprintf(1,'Movie %d: %s\n',imov,mov);
                
        parfor i=idx(:)'
          s = sloc(i);
          imfrmf = fullfile(packdir,sdir,[s.idmovfrm '.png']);
          if exist(imfrmf,'file')>0
            fprintf(1,'Skipping, image already exists: %s\n',imfrmf);
          else
            % calling get_readframe_fcn outside parfor results in harderr
            rfcn = get_readframe_fcn(mov);
            imfrm = rfcn(s.frm);
            imwrite(imfrm,imfrmf);
            fprintf(1,'Wrote %s\n',imfrmf);
          end
        end
      end
    end
%     function writeimscc(sloccc,packdir)
%       
%       SUBDIRIM = 'imcc';
%       sdir = SUBDIRIM;
%       if exist(fullfile(packdir,sdir),'dir')==0
%         mkdir(packdir,sdir);
%       end
%       
%       
%       mr = MovieReader;
%       for i=1:numel(sloccc)
%         s = sloccc(i);
%         
%         % Expect sloc to be in 'movie order'
%         if ~strcmp(s.mov,mr.filename)
%           mr.close();
%           mr.open(s.mov);
%           fprintf(1,'Opened movie: %s\n',s.mov);
%         end
%       
%         imfrm = mr.readframe(f);
%         imfrmmask = imfrm;
%         imfrmmask(~maskcc) = 0;
%         imfrmmaskcrop = imfrmmask(r0:r1,c0:c1);
%         if writeims
%           basefSpng = [basefS '.png'];
%           basefSimfrmpng = [basefSimfrm '.png'];
%           %maskf = fullfile(packdir,SUBDIRMASK,basefSpng);
%           imfrmf = fullfile(packdir,SUBDIRIM,basefSimfrmpng);
%           %imfrmmaskf = fullfile(packdir,SUBDIRIMMASK,basefSpng);
%           imfrmmaskcropf = fullfile(packdir,SUBDIRIMMASKC,basefSpng);
%           
%           %imwrite(mask,maskf);
%           if icc==1
%             imwrite(imfrm,imfrmf);
%           end
%           %imwrite(imfrmmask,imfrmmaskf);
%           imwrite(imfrmmaskcrop,imfrmmaskcropf);
%           fprintf(1,'Wrote files for %s...\n',basefS);s
%         else
%           fprintf(1,'Didnt write files for %s...\n',basefS);
%         end
%       end
%     end
  end
  methods (Static) % stripped lbl
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
    function s = compressStrippedLbl(s,varargin)
      isMA = s.cfg.MultiAnimal;
      
      CFG_GLOBS = {'Num' 'MultiAnimal'};
      FLDS = {'cfg' 'projname' 'projectFile' 'projMacros' 'movieInfoAll' 'cropProjHasCrops' ...
        'trackerClass' 'trackerData'};
      TRACKERDATA_FLDS = {'sPrmAll' 'trnNetTypeString'};
      if isMA
        GLOBS = {'movieFilesAll' 'trxFilesAll'};
        FLDSRM = {'projMacros'};
      else
        GLOBS = {'labeledpos' 'movieFilesAll' 'trxFilesAll' 'preProcData'};
        FLDSRM = { ... % 'movieFilesAllCropInfo' 'movieFilesAllGTCropInfo' ...
                  'movieFilesAllHistEqLUT' 'movieFilesAllGTHistEqLUT'};
      end
      
      fldscfg = fieldnames(s.cfg);      
      fldscfgkeep = fldscfg(startsWith(fldscfg,CFG_GLOBS));
      s.cfg = structrestrictflds(s.cfg,fldscfgkeep);
      
      s.trackerData{2} = structrestrictflds(s.trackerData{2},TRACKERDATA_FLDS);
      
      flds = fieldnames(s);
      fldskeep = flds(startsWith(flds,GLOBS));
      fldskeep = [fldskeep(:); FLDS(:)];
      fldskeep = setdiff(fldskeep, FLDSRM);
      s = structrestrictflds(s,fldskeep);
    end
    function [jse,j] = jsonifyStrippedLbl(s)
      % s: compressed stripped lbl (output of compressStrippedLbl)
      %
      % jse: jsonencoded struct
      % j: raw struct
      
      cfg = s.cfg;
      cfg.HasCrops = s.cropProjHasCrops;
      mia = cellfun(@(x)struct('NumRows',x.info.nr,...
                               'NumCols',x.info.nc),s.movieInfoAll);
      for ivw=1:size(mia,2)
        nr = [mia(:,ivw).NumRows];
        nc = [mia(:,ivw).NumCols];
        assert(all(nr==nr(1) & nc==nc(1)),'Inconsistent movie dimensions for view %d',ivw);        
      end
      
      j = struct();
      j.ProjName = s.projname;
      j.Config = cfg;
      j.MovieInfo = mia(1,:);
      assert(strcmp(s.trackerClass{2},'DeepTracker'));
      j.TrackerData = s.trackerData{2};
      
      jse = jsonencode(j);
    end
  end
end