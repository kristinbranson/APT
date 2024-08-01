classdef TrnPack
  % DL Training Package 

  properties (Constant)
    SUBDIRIM = 'im';
  end
  methods (Static)
    
    function vizLoc(packdir,varargin)
      % Visualize 'loc' data structure (one row per labeled mov,frm,tgt) 
      % from training package.
      %
      % packdir: package dir (contains images)

      [scargs,ttlargs,loc] = myparse(varargin,...
        'scargs',{16}, ...
        'ttlargs',{'fontsize',16,'fontweight','bold','interpreter','none'}, ...
        'loc',[] ... % (opt), specific loc struct array to show
        );
      
      [~,~,~,loc] = TrnPack.loadPack(packdir);      
      loc = loc.locdata;
      
      nvw = numel(loc(1).img);
      hfig = figure(11);      
      axs = createsubplots(1,nvw);
      
      idmfun = unique({loc.idmovfrm}');
      nmfun = numel(idmfun);
      for iidmf=1:nmfun
        idmf = idmfun{iidmf};
        is = find(strcmp({loc.idmovfrm}',idmf));
        
        for ivw=1:nvw
          imfS = loc(is(1)).img{ivw};
          imf = fullfile(packdir,imfS);
          if exist(imf,'file')==0
            warningNoTrace('Skipping, image not found: %s',imf);
            continue;
          end

          im = imread(imf);

          axes(axs(ivw));
          cla;          
          imagesc(im);
          colormap gray;
          hold on;
          axis square;

          for j = is(:)'
            s = loc(j);
            xy = reshape(s.pabs,[],2);
            npts = size(xy,1)/nvw;
            iptsvw = (ivw-1)*npts + (1:npts)';            
            scatter(xy(iptsvw,1),xy(iptsvw,2),scargs{:});        
            plot(s.roi([1:4 1]),s.roi([5:8 5]),'r-','linewidth',2);
          end

          if ivw==1
            tstr = sprintf('%s: %d tgts',idmf,numel(is));
            title(tstr,ttlargs{:});
          end
        end
        
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
        [~,~,~,locg] = TrnPack.loadPack(packdir);
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
      
      [~,~,~,~,~,loccc] = TrnPack.loadPack(packdir); % xxx api now broken

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
    function nlbls = readNLabels(tpjson)
       tp = TrnPack.hlpLoadJson(tpjson);
       nlbls = arrayfun(@(x)size(x.p,2),tp);
    end
    function [slbl,j,tp,locg] = loadPack(packdir,varargin)
      % Load training package into MATLAB data structures
      %
      % slbl: 'stripped lbl' struct
      % j: cfg/json
      % tp: one-row-per-movie struct. Maybe a useful format for metadata 
      %   or bookkeeping purposes.
      % loc: one-row-per-labeled-(mov,frm,tgt) struct. Intended to be
      %   primary MA keypt data structure.
      % loccc: one-row-per-labeled-cluster. Experimental, may not be
      %   useful for DL/py backend.
      %
      % Note tp, loc, loccc contain equivalent info just in different
      % formats.
      
      if ~DeepModelChainOnDisk.gen_strippedlblfile,
        error('This code will not run without the stripped lbl file. Tell Kristin how you got here.');
      end

      incTrnPack = myparse(varargin,...
        'incTrnPack',false ...
        );
       
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
      
      [~,f,~] = fileparts(lblsf);
      jf = fullfile(packdir,[f '.json']);
      fprintf(1,'loaded %s\n',jf);
      jf = readtxtfile(jf);
      j = jsondecode(jf{1});

      if incTrnPack
        tpf = fullfile(packdir,'trnpack.json');
        tp = TrnPack.hlpLoadJson(tpf);
      else
        tp = [];
      end

%       locf = fullfile(packdir,'loc0.json');
%       loc = TrnPack.hlpLoadJson(locf);

      locf = fullfile(packdir,'loc.json');
      locg = TrnPack.hlpLoadJson(locf);

%       locf = fullfile(packdir,'locclus.json');
%       if exist(locf,'file')>0
%         locjse = readtxtfile(locf);
%         loccc = jsondecode(locjse{1});
%         fprintf(1,'loaded %s\n',locf);
%       else
%         loccc = [];
%       end
    end
    
    function hlpSaveJson(s,jsonoutf)
      j = jsonencode(s,'ConvertInfAndNaN',false);
      %jsonoutf = fullfile(packdir,jsonoutf);
      fh = fopen(jsonoutf,'w');
      fprintf(fh,'%s\n',j);
      fclose(fh);
      fprintf(1,'Wrote %s.\n',jsonoutf);
    end
    function [slbl,tp,locg,ntgtstot] = genWriteTrnPack(lObj,dmc,varargin)
      % Generate training package. Write contents (raw images and keypt 
      % jsons) to packdir.
      
      [writeims,writeimsidx,trainConfigName,slblname,verbosejson,tblsplit,view] = myparse(varargin,...
        'writeims',true, ...
        'writeimsidx',[], ... % (opt) DEBUG ONLY
        'trainConfigName','',...
        'strippedlblname',[], ... % (reqd) short filename for stripped lbl
        'verbosejson',false, ...
        'tblsplit', [], ...  % tbl with fields .mov, .frm, .split
                       ...  % all double/numeric and 1-based
        'view',nan ... % currently unused
        );      
      
      tfsplitsprovided = ~isempty(tblsplit);

      packdir = dmc.dirProjLnx;
      
      if exist(packdir,'dir')==0
        mkdir(packdir);
      end
      
      tObj = lObj.tracker;
      tObj.setAllParams(lObj.trackGetParams()); % does not set skel, flipLMEdges

      slbl_orig = tObj.trnCreateStrippedLbl();
      slbl = Lbl.compressStrippedLbl(slbl_orig,'ma',true);
      [~,jslbl] = Lbl.jsonifyStrippedLbl(slbl);

      if strcmp(DeepModelChainOnDisk.configFileExt,'.lbl') || DeepModelChainOnDisk.gen_strippedlblfile,
        sfname = dmc.lblStrippedLnx;
%         assert(~isempty(slblname));
%         sfname = fullfile(packdir,slblname);
        save(sfname,'-mat','-v7.3','-struct','slbl');
        fprintf(1,'Saved %s\n',sfname);
      end
      
      if strcmp(DeepModelChainOnDisk.configFileExt,'.json'),
        TrnPack.hlpSaveJson(jslbl,DeepModelChainOnDisk.getCheckSingle(dmc.trainConfigLnx));
      end

      % use stripped lbl trackerData instead of tObj, as we have called
      % addExtraParams etc.
      
      tdata2 = slbl.trackerData{2};  
      netmode2 = tdata2.trnNetMode;
      % For top-down nets, whether trnNetMode from .trackerData{2} or 
      % .trackerData{1} is used should be irrelevant to isObjDet.
      isObjDet = netmode2.isObjDet;
      % Again, getting sPrmBBox and sPrmLossMask from trackerData{2} should
      % be fine for top-down nets. trackerData{1} should match for these
      % params.
      sPrmMA = tdata2.sPrmAll.ROOT.MultiAnimal;
      sPrmBBox = sPrmMA.Detect.BBox; 
      sPrmLossMask = sPrmMA.LossMask;
      tp = TrnPack.aggregateLabelsAddRoi(lObj,isObjDet,sPrmBBox,sPrmLossMask);
      
      % add splits to tp
      nmov = numel(tp);
      if tfsplitsprovided

        % split verification
        tblsplitMF = tblsplit(:,{'mov' 'frm'});
        assert(height(tblsplitMF)==height(unique(tblsplitMF)),...
          'Split table contains one or more duplicated (movie,frame) pairs.');
        allsplits = tblsplit.split;
        assert(isequal(round(allsplits),allsplits));
        assert(all(allsplits>0));
        nsplits = max(allsplits);
        assert(isequal(unique(allsplits),(1:nsplits)'));
        
        tfsplitrowused = false(height(tblsplit),1);
        for imov=1:nmov
          tfmov = tblsplit.mov==imov;
          idxmov = find(tfmov);
          fsplit = tblsplit.frm(tfmov);
          jsplit = tblsplit.split(tfmov);
          
          [tf,loc] = ismember(tp(imov).frm,fsplit);
          split = zeros(size(tp(imov).frm),Labels.CLS_SPLIT);
          nMissing = nnz(~tf);
          if nMissing>0
            warningNoTrace('Movie %d: %d labeled rows not found in split table. These will be added to split 1.',...
              imov,nMissing);
            split(~tf) = 1;
          end          
          
          split(tf) = jsplit(loc);          
          tfsplitrowused(idxmov(loc)) = true;
          tp(imov).split = split;
        end
        
        nExtra = nnz(~tfsplitrowused);
        if nExtra>0
          warningNoTrace('Ignoring %d unused/extraneous rows in split table.',nExtra);
        end
        
        fprintf(1,'Split summary:\n');
        summary(categorical(allsplits(tfsplitrowused)));
      else
        for imov=1:nmov
          tp(imov).split = ones(size(tp(imov).frm),Labels.CLS_SPLIT);
        end
      end
      
      if lObj.gtIsGTMode
        movinfo = lObj.movieInfoAllGT;
      else
        movinfo = lObj.movieInfoAll;
      end
      isma = lObj.maIsMA;
      if isma
        locg = TrnPack.genLocs(tp,movinfo);
        if writeims
          if isempty(writeimsidx)
            writeimsidx = 1:numel(locg);
          end

          TrnPack.writeims(locg(writeimsidx),packdir);
        end
      else
        locg = TrnPack.genLocsSA(slbl_orig,tblsplit);
        if writeims
          if isempty(writeimsidx)
            writeimsidx = 1:numel(locg);
          end

          TrnPack.writeimsSA(locg(writeimsidx),packdir,slbl_orig.preProcData_I);
        end
      end
        
      if verbosejson
        % trnpack: one row per mov
        %jsonoutf = 'trnpack.json';
        jsonoutf = dmc.trainPackLnx;
        TrnPack.hlpSaveJson(tp,jsonoutf);
      end
      
%       % loc: one row per labeled tgt
%       jsonoutf = 'loc0.json';
%       TrnPack.hlpSaveJson(loc,packdir,jsonoutf);

      % loc: one row per frm
      jsonoutf = dmc.trainLocLnx;
      %jsonoutf = 'loc.json';
      s = struct();
      s.movies = lObj.movieFilesAllFull;
      if tfsplitsprovided
        s.splitnames = arrayfun(@(x)sprintf('split%02d',x),1:nsplits,'uni',0);
      else
        s.splitnames = {'trn'};
      end
      s.locdata = locg;
      TrnPack.hlpSaveJson(s,jsonoutf);
      
      ntgtstot = sum([locg.ntgt]);

%       % loccc: one row per cluster
%       jsonoutf = 'locclus.json';
%       TrnPack.hlpSaveJson(loccc,packdir,jsonoutf);      
    end
    
    function sagg = aggregateLabelsAddRoi(lObj,isObjDet,sPrmBBox,...
        sPrmLossMask,varargin)
      
      [incPartialRows,treatInfPosAsOcc] = myparse(varargin,...
        'incPartialRows',false,...
        'treatInfPosAsOcc',true ...
        );
      
      isgt = lObj.gtIsGTMode;
      PROPS = lObj.gtGetSharedProps;
      fLbl = PROPS.LBL;
      fmfaf = PROPS.MFAF;
      
      lbls = lObj.(fLbl);
      mfafs = lObj.(fmfaf);
      nmov = numel(lbls);
      sagg = cell(nmov,1);
      for imov=1:nmov
        s = lbls{imov};
        s.mov = mfafs{imov};
        
        % see also from Labeler/preProcGetMFTableLbled      
        if ~incPartialRows
          s = Labels.rmRows(s,@isnan,'partially-labeled');
        end
        if treatInfPosAsOcc
          s = Labels.replaceInfWithNan(s);
        end
        
        %% gen rois, bw
        n = size(s.p,2);
        s.roi = nan(8,n);
%        fprintf(1,'mov %d: %d labeled frms.\n',imov,n);
        for i=1:n
          p = s.p(:,i);
          xy = Shape.vec2xy(p);
          if isObjDet
            minaa = sPrmBBox.MinAspectRatio;
            roi = lObj.maComputeBboxGeneral(xy,minaa,false,[],[]);
          else
            roi = lObj.maGetLossMask(xy,sPrmLossMask);
          end
          s.roi(:,i) = roi(:);
        end

        if ~isgt
          sroi = lObj.labelsRoi{imov};
        else
          sroi = LabelROI.new();
        end
        s.frmroi = sroi.f;
        s.extra_roi = sroi.verts;
        sagg{imov} = s;        
      end
      sagg = cell2mat(sagg);
    end
    function slocg = genLocs(sagg,movInfoAll)
      assert(numel(sagg)==numel(movInfoAll));
      nmov = numel(sagg);
      %sloc = [];
      slocg = [];
      %sloccc = [];
      for imov=1:nmov
        s = sagg(imov);
        movifo = movInfoAll{imov};
        imsz = [movifo.info.nr movifo.info.nc];
        fprintf(1,'mov %d (sz=%s): %s\n',imov,mat2str(imsz),s.mov);
        
        %slocI = TrnPack.genLocsI(s,imov);
        slocgI = TrnPack.genLocsGroupedI(s,imov);
        %slocccI = TrnPack.genCropClusteredLocsI(s,imsz,imov);
        
        %sloc = [sloc; slocI]; %#ok<AGROW>
        slocg = [slocg; slocgI]; %#ok<AGROW>
        %sloccc = [sloccc; slocccI]; %#ok<AGROW>
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
      
%       s = Labels.addsplitsifnec(s);

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
          'idmovfrm',sprintf('mov%04d_frm%08d',imov,f),...
          'img',{{img}},...
          'imov',imov,... 
          'mov',s.mov,...
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
    function [sloc] = genLocsSA(slbl,tblsplit,varargin)
    % Locs for Single animal. Use images in the lbl cache for now.
    % mov is empty for now. Seemed too convoluted to include it for now.
    % MK 07022022
      imgpat = myparse(varargin,...
        'imgpat','im/%s.png' ...
        );
      
      nrows=size(slbl.preProcData_I,1);
      sloc = [];
      roi = [];
      for v=1:size(slbl.preProcData_I,2)
        c1 = size(slbl.preProcData_I{v},2);
        r1 = size(slbl.preProcData_I{v},1);
        cur_roi = [1 1 c1 c1 1 r1 r1 1]';
        roi = [roi; cur_roi];
      end
      has_split = ~isempty(tblsplit);
      default_split = 1;
      
      for j=1:nrows
        f = slbl.preProcData_MD_frm(j);
        itgt = slbl.preProcData_MD_iTgt(j);
        ts = slbl.preProcDataTS;
        occ = slbl.preProcData_MD_tfocc(j,:);
        imov = slbl.preProcData_MD_mov(j);
        if has_split
          curndx = find((tblsplit.mov == imov) & (tblsplit.frm ==f) & ...
            (tblsplit.iTgt == itgt));
          assert(numel(curndx)==1);
          split = tblsplit(curndx,:).split;
        else          
          split = default_split;
        end
        
        imgs = {};
        for v=1:slbl.cfg.NumViews
          basefS = sprintf('mov%04d_frm%08d_tgt%05d_view%d',imov,f,itgt,v);
          img = sprintf(imgpat,basefS);
          imgs{v} = img;
        end
        sloctmp = struct(...
          'id',sprintf('mov%04d_frm%08d_tgt%05d',imov,f,itgt),...
          'idmovfrm',sprintf('mov%04d_frm%08d_tgt%05d',imov,f,itgt),...
          'img',{imgs},...
          'imov',imov,...
          'mov','',...
          'frm',f,...
          'itgt',itgt,...
          'ntgt',1,...
          'roi',roi,...
          'split',split,...
          'pabs',slbl.preProcData_P(j,:), ...
          'occ',occ, ...
          'ts',ts ...
          );
        sloc = [sloc; sloctmp]; %#ok<AGROW>
      end
    end

    function writeims(sloc,packdir)
      % Currently single-view only
      
      sdir = TrnPack.SUBDIRIM;
      if exist(fullfile(packdir,sdir),'dir')==0
        mkdir(packdir,sdir);
      end
      
      imovall = [sloc.imov]';
      imovun = unique(imovall);
      
      fprintf(1,'Writing training images...\n');
      
      bufsize = 128;      
      for iimov = 1:numel(imovun),
        imov=imovun(iimov);
        idx = find(imovall==imov); % indices into sloc for this mov
        % idx cannot be empty
        mov = sloc(idx(1)).mov;
        %mr.open(mov);
        fprintf(1,'Movie %d: %s (%d/%d)\n',imov,mov,iimov,numel(imovun));
        [rfcn,~,fid] = get_readframe_fcn(mov);
        frms = [sloc(idx).frm];
        filenames = arrayfun(@(i) fullfile(packdir,sdir,[sloc(i).idmovfrm '.png']),idx,'Uni',0);
        doskip = cellfun(@(x) exist(x,'file'),filenames) > 0;
        if ~all(doskip),
          curfilenames = filenames(~doskip);
          res = parforOverVideo(rfcn,frms(~doskip),@(im,frm,i) imwriteCheck(im,curfilenames{i}),'bufsize',bufsize,'verbose',true);
          assert(all(cell2mat(res)));
        end
        fprintf(1,'Wrote %d new images, %d existed previously\n',nnz(~doskip),nnz(doskip));

        if ~isempty(fid) && fid > 1,
          fclose(fid);
        end
      end
      
    end
    function writeimsSA(sloc,packdir,ims)
      % Write ims for Single animal
            
      sdir = TrnPack.SUBDIRIM;
      if exist(fullfile(packdir,sdir),'dir')==0
        mkdir(packdir,sdir);
      end
      
      n=numel(sloc);
      fprintf(1,'Writing %d training images...\n',n);

      parfor i=1:n
        imgnames = sloc(i).img;
        for v=1:size(ims,2)
          imgfile = fullfile(packdir,imgnames{v});
          im = ims{i,v};
          if ~exist(imgfile,'file'),
            imwrite(im,imgfile);
          end
        end
      end
    end

    
    function clearims(packdir)
      sdir = TrnPack.SUBDIRIM;
      imdir = fullfile(packdir,sdir);
      if exist(imdir,'dir')==0
        return;
      end
      [succ,msg,mid] = rmdir(imdir,'s');
      if ~succ
        error(mid,'Failed to clear image cache: %s',msg);
      end
    end

    function t = toMFT(tp)
      mov = zeros(0,1);
      frm = mov;
      iTgt = mov;
      for imov=1:numel(tp)
        ntgt = numel(tp(imov).frm);
        mov = cat(1,mov,repmat(imov,ntgt,1));
        frm = cat(1,frm,tp(imov).frm);
        iTgt = cat(1,iTgt,tp(imov).tgt);
      end
      t = table(mov,frm,iTgt);
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
  
end