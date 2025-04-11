function maErr = computeMAErr(tblPred,tblLbl)
  
  nfrm = size(tblLbl,1);
  mov_frm_pred = [tblPred.mov tblPred.frm];
  mov_frm_lbl =  [tblLbl.mov tblLbl.frm];
  nlbl = size(mov_frm_lbl,1);
  psz = size(tblPred.pTrk);
  npts = psz(end)/2;
  maxn = psz(2);

  [~,match] = ismember(mov_frm_pred,mov_frm_lbl,'rows');

  all_lbl = reshape(tblLbl.p,[nlbl,maxn,npts,2]);
  all_pred = reshape(tblPred.pTrk,[size(tblPred,1),maxn,npts,2]);

  maErr = nan(nfrm,maxn,npts);  
  for ndx = 1:nfrm
    elbl = []; epreds = [];    
    epreds(1,:,:,:) = all_pred(ndx,:,:,:);
    elbl(:,1,:,:) = all_lbl(match(ndx),:,:,:);

    epreds = repmat(epreds,[size(elbl,1),1]);
    elbl = repmat(elbl,[1,size(epreds,2)]);
    dist_mat = sqrt(sum( (epreds-elbl).^2,4));
    dist_match = find_dist_match(dist_mat);
    maErr(ndx,:,:) = dist_match;
  end  
end


function dout = find_dist_match(dd)
    dout = nan(size(dd(:,1,:)));
    yy = mean(dd,3,'omitnan');
    sel2 = any(~isnan(yy),1);
    sel1 = any(~isnan(yy),2);
    cmat = yy(sel1,:);
    cmat = cmat(:,sel2);
    idx_match = matchpairs(cmat,max(cmat(:))*10);
%     [idx1, idx2] = munkres(cmat);
    sel_ndx1 = find(sel1);
    sel_ndx2 = find(sel2);
    orig_idx1 = sel_ndx1(idx_match(:,1));
    orig_idx2 = sel_ndx2(idx_match(:,2));
    for ix = 1:size(idx_match,1)
        dout(orig_idx1(ix),:) = dd(orig_idx1(ix),orig_idx2(ix),:);
    end
end

function [dout, oo_out] = find_dist_match_occ(dd,oo)
    dout = nan(size(dd(:,:,1,:)));
    oo_out = zeros(size(dout));
    yy = nanmean(dd,4);
    for a = 1:size(dd,1)
        for ndx = 1:size(dd,3)
            if all(isnan(yy(a,:,ndx)))
                continue
            end
            r = find(yy(a,:,ndx) == min(yy(a,:,ndx)),1);
            dout(a,ndx,:) = dd(a,r,ndx,:);
            oo_out(a,ndx,:) = oo(a,r,:);
        end
    end
end
