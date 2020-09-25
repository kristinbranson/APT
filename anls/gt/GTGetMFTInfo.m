function gtdata = GTGetMFTInfo(lObj,gtdata,vwi)

if ~exist('vwi','var'),
  vwi = 1;
end

if isstruct(lObj),
  labeledpos = cellfun(@SparseLabelArray.full,lObj.labeledposGT,'uni',0);
else
  labeledpos = lObj.labeledpos;
end

nets = fieldnames(gtdata);
nnets = numel(nets);

for neti = 1:nnets,
  
  net = nets{neti};
  
  for ndx = 1:numel(gtdata.(net)),
    
    [ndata,nlandmarks,d] = size(gtdata.(net){ndx}.labels);
    nvws = size(labeledpos{1},1)/nlandmarks;
    
    for i = 1:numel(labeledpos),
      sz = size(labeledpos{i});
      assert(sz(1)==nlandmarks*nvws);
      labeledpos{i} = reshape(labeledpos{i},[nlandmarks,nvws,sz(2:end)]);
      labeledpos{i} = reshape(labeledpos{i}(:,vwi,:,:),[nlandmarks,sz(2:end)]);
    end
    alllabels = nan([nlandmarks,d,0]);
    allmfts = nan(3,0);
    for i = 1:numel(labeledpos),
      nfrs = size(labeledpos{i},3);
      ntgts = size(labeledpos{i},4);
      idx = find(~isnan(labeledpos{i}(1,1,:,:)));
      if isempty(idx),
        continue;
      end
      [fs,tgts] = ind2sub([nfrs,ntgts],idx);
      n = numel(idx);
      alllabels(:,:,end+1:end+n) = labeledpos{i}(:,:,idx);
      allmfts(:,end+1:end+n) = cat(1,zeros(1,n)+i,fs(:)',tgts(:)');
    end
    
    alllabels = permute(alllabels,[3,1,2]);
    alllabels0 = alllabels-alllabels(:,1,:);
    netlabels = gtdata.(net){ndx}.labels;
    netlabels0 = netlabels-netlabels(:,1,:);
    isinorder = false;
    if size(alllabels0,1) == size(netlabels0,1),
      
      d = sum(sum(abs(alllabels0-netlabels0),2),3);
      if max(d) < .01,
        order = 1:ndata;
        isinorder = true;
      end
      
    end
    
    if ~isinorder,
      
      order = nan(1,size(netlabels0,1));
      for i = 1:size(netlabels0,1),
        if ndata >= i && sum(sum(abs(netlabels0(i,:,:)-alllabels0(i,:,:)),2),3) < .01,
          order(i) = i;
        else
          d = sum(sum(abs(netlabels0(i,:,:)-alllabels0),2),3);
          orderprev = order(1:i-1);
          orderprev(isnan(orderprev)) = [];
          d(orderprev) = nan;
          [mind,j] = min(d);
          if mind > .01,
            fprintf('No match found for i = %d, m = %d, f = %d, t = %d\n',i,allmfts(:,i));
            if ndata >= i && ~any(order(1:i-1)==i),
              fprintf('Setting match to %d anyways\n',i);
              order(i) = i;
            end
          else
            order(i) = j;
          end
        end
      end
      
    end
    isinorder = all(diff(order)==1);
    fprintf('Net %s, model %d is in order = %d\n',net,ndx,isinorder);
    gtdata.(net){ndx}.mov = allmfts(1,:);
    gtdata.(net){ndx}.frm = allmfts(2,:);
    gtdata.(net){ndx}.iTgt = allmfts(3,:);
    
  end
end
