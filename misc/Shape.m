classdef Shape 

  methods
  
    function p0 = randrot(p0,d)
      % Randomly rotate shapes
      % 
      % p0 (in): [Lx2d] shapes
      % d: model.d. Must be 2.
      %
      % p0 (out): [Lx2d] randomly rotated shapes (each row of p0 randomly
      %   rotated)
      
      assert(d==2);
      [L,D] = size(p0);      
      nfids = D/d;
      
      thetas = 2*pi*rand(L,1);
      ct = cos(thetas); % [Lx1]
      st = sin(thetas); % [Lx1]
      p0 = reshape(p0,[L,nfids,d]);
      mus = mean(p0,2); % [Lx1x2] centroids
      p0 = bsxfun(@minus,p0,mus);
      x = bsxfun(@times,ct,p0(:,:,1)) - bsxfun(@times,st,p0(:,:,2)); % [Lxnfids]
      y = bsxfun(@times,st,p0(:,:,1)) + bsxfun(@times,ct,p0(:,:,2)); % [Lxnfids]
      p0 = cat(3,x,y); % [Lxnfidsx2]
      p0 = bsxfun(@plus,p0,mus);
      p0 = reshape(p0,[L,nfids*d]);
    end
    
    function p1 = randsamp(p0,i,L)
      % Randomly sample shapes
      %
      % p0 (in): [NxD] all shapes
      % i: index of 'current' shape (1..L)
      % L: number of shapes to return
      %
      % p1: [LxD] shapes randomly sampled from p0, *omitting* the nth
      %   shape, ie p0(n,:).
            
      [N,D] = size(p0);
      assert(any(i==1:N));      
      
      iOther = [1:i-1,i+1:N];
      if L <= N-1
        i1 = randSample(iOther,L,true); %[n randSample(1:N,L-1)];
        p1 = p0(i1,:);
      else % L>N-1
        % Not enough other shapes. Select pairs of shapes and average them
        % AL: min() seems unnecessary if use (N-1)*rand(...)
        nExtra = L-(N-1);
        iAv = iOther(ceil((N-1)*rand([nExtra,2]))); % [nExtrax2] random elements of iOther
        pAv = (p0(iAv(:,1),:) + p0(iAv(:,2),:))/2; % [nExtraxD] randomly averaged shapes
        p1 = cat(1,p0(iOther,:),pAv);
      end
      
      % p1: Set of L shapes, explicitly doesn't include p0(n,:)
      assert(isequal(size(p1),[L D]));      
    end
    
    function bbJ = jitterBbox(bb,L,d,uncertfac)
      % Randomly jitter bounding box
      % bb: [1x2d] bounding box [x1 x2 .. xd w1 w2 .. wd]
      % L: number of replicates
      % d: dimension
      % uncertfac: scalar double
      
      assert(isequal(size(bb),[1 2*d]));
      szs = bb(d+1:end);
      maxDisp = szs/uncertfac;
      uncert = bsxfun(@times,(2*rand(L,d)-1),maxDisp);
      
      bbJ = repmat(bb,[L,1]);
      bbJ(:,1:d) = bbJ(:,1:d) + uncert;
    end
    
  end
  
end
