function [maErr,fp,fn] = computeMAErr(tblPred,tblLbl,hasMask)
  
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
  fp = nan(nfrm,maxn);
  fn = nan(nfrm,maxn);
  for ndx = 1:nfrm
    elbl = []; epreds = [];    
    epreds1(1,:,:,:) = all_pred(ndx,:,:,:);
    elbl1(:,1,:,:) = all_lbl(match(ndx),:,:,:);

    epreds = repmat(epreds1,[size(elbl1,1),1]);
    elbl = repmat(elbl1,[1,size(epreds1,2)]);
    dist_mat = sqrt(sum( (epreds-elbl).^2,4));
    [dist_match,fp_cur,fn_cur] = find_dist_match(dist_mat,hasMask,elbl1,epreds1);
    maErr(ndx,:,:) = dist_match;
    fp(ndx,:) = fp_cur;
    fn(ndx,:) = fn_cur;
  end  
end


function [dout,fp,fn] = find_dist_match(dd,hasMask,elbl,epreds)
    dout = nan(size(dd(:,1,:)));
    fp = nan(size(dd(1,:,1)));
    fn = nan(size(dd(:,1,1)));
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
        fp(orig_idx2(ix)) = 0;
        fn(orig_idx1(ix)) = 0;
    end

    for xx = sel_ndx1(:)'
      if all(xx~=orig_idx1)
        fn(xx) = 1;
      end
    end

    if hasMask
      elbl = permute(elbl,[1,3,4,2]);
      epreds = permute(epreds,[2,3,4,1]);
      bbox_lbl_min = min(elbl,[],2,"omitmissing");
      bbox_lbl_max = max(elbl,[],2,"omitmissing");
      bbox_lbl = cat(3,bbox_lbl_min,bbox_lbl_max);
      bbox_lbl = permute(bbox_lbl,[1,3,2]);
      bbox_pred_min = min(epreds,[],2,"omitmissing");
      bbox_pred_max = max(epreds,[],2,"omitmissing");
      bbox_pred = cat(3,bbox_pred_min,bbox_pred_max);
      bbox_pred = permute(bbox_pred,[1,3,2]);

      overlap = bboxOverlapMatrix(bbox_lbl,bbox_pred);
      for xx = sel_ndx2(:)'
        if all(xx~=orig_idx2) && any(overlap(:,xx)>0.2)
            fp(xx) = 1;
        end
      end
    else
      for xx = sel_ndx2(:)'
        if all(xx~=orig_idx2)
          fp(xx) = 1;
        end
      end

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

function overlap_ratio = bboxOverlapRatio(bbox1, bbox2)
    % Compute overlap ratio between two bounding boxes
    % Input: bbox1, bbox2 are [x, y, width, height] or [x1, y1, x2, y2]
    % Output: overlap_ratio (0 to 1, where 1 is complete overlap)
    
    % Find intersection coordinates
    x1 = max(bbox1(1), bbox2(1));
    y1 = max(bbox1(2), bbox2(2));
    x2 = min(bbox1(3), bbox2(3));
    y2 = min(bbox1(4), bbox2(4));
    
    % Compute intersection area
    if x2 > x1 && y2 > y1
        intersection_area = (x2 - x1) * (y2 - y1);
    else
        intersection_area = 0;
    end
    
    % Compute areas of both boxes
    area1 = (bbox1(3) - bbox1(1)) * (bbox1(4) - bbox1(2));
    area2 = (bbox2(3) - bbox2(1)) * (bbox2(4) - bbox2(2));
    
    % Compute union area
    union_area = area1 + area2 - intersection_area;
    
    % Compute IoU (Intersection over Union)
    if union_area > 0
        overlap_ratio = intersection_area / union_area;
    else
        overlap_ratio = 0;
    end
end

function overlap_area = bboxOverlapArea(bbox1, bbox2)
    % Compute absolute overlap area between two bounding boxes
    % Input: bbox1, bbox2 are [x, y, width, height] or [x1, y1, x2, y2]
    % Output: overlap_area in pixels
    
    % Find intersection coordinates
    x1 = max(bbox1(1), bbox2(1));
    y1 = max(bbox1(2), bbox2(2));
    x2 = min(bbox1(3), bbox2(3));
    y2 = min(bbox1(4), bbox2(4));
    
    % Compute intersection area
    if x2 > x1 && y2 > y1
        overlap_area = (x2 - x1) * (y2 - y1);
    else
        overlap_area = 0;
    end
end

function overlap_matrix = bboxOverlapMatrix(bboxes1, bboxes2)
    % Compute overlap ratios between two sets of bounding boxes
    % Input: bboxes1 (N x 4), bboxes2 (M x 4)
    % Output: overlap_matrix (N x M) with IoU values
    
    n1 = size(bboxes1, 1);
    n2 = size(bboxes2, 1);
    overlap_matrix = zeros(n1, n2);
    
    for i = 1:n1
        for j = 1:n2
            overlap_matrix(i, j) = bboxOverlapRatio(bboxes1(i, :), bboxes2(j, :));
        end
    end
end

function is_overlapping = bboxIsOverlapping(bbox1, bbox2, threshold)
    % Check if two bounding boxes overlap above a threshold
    % Input: bbox1, bbox2, threshold (default 0.5)
    % Output: is_overlapping (logical)
    
    if nargin < 3
        threshold = 0.5;
    end
    
    overlap_ratio = bboxOverlapRatio(bbox1, bbox2);
    is_overlapping = overlap_ratio >= threshold;
end

% Example usage:
% bbox1 = [10, 10, 50, 30];  % [x, y, width, height]
% bbox2 = [30, 20, 40, 25];
% 
% overlap_ratio = bboxOverlapRatio(bbox1, bbox2);
% overlap_area = bboxOverlapArea(bbox1, bbox2);
% is_overlapping = bboxIsOverlapping(bbox1, bbox2, 0.3);
% 
% fprintf('Overlap ratio (IoU): %.3f\n', overlap_ratio);
% fprintf('Overlap area: %.1f pixels\n', overlap_area);
% fprintf('Is overlapping (>0.3): %d\n', is_overlapping);