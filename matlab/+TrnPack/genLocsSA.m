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
    roi = [roi; cur_roi];  %#ok<AGROW>
  end
  has_split = ~isempty(tblsplit);
  default_split = 1;

  for j=1:nrows
    f = slbl.preProcData_MD_frm(j);
    itgt = slbl.preProcData_MD_iTgt(j);
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
      imgs{v} = img;  %#ok<AGROW>
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
      'occ',occ ...
      );
    sloc = [sloc; sloctmp]; %#ok<AGROW>
  end
end % function
