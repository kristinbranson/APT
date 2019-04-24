function displayimages(I,nr,nc,label)

nI = numel(I);
nperpage = nr*nc;
npages = ceil(nI/nperpage);

for page = 1:npages,

  clf;
  
  istart = (page-1)*nperpage;
  iend = min( page*nperpage, nI );
  ni = iend - istart;

  for i = 1:ni,

    subplot(nr,nc,i);
    imshow(I{istart+i});

    if exist('label')
      if isnumeric(label),
        title(num2str(label(istart+i)));
      elseif isstr(label(istart+i,:)),
        title(label(istart+i,:));
      end;
    end;

  end;

  fprintf('Page %d / %d.\n',page,npages);
  if ~exist('label')
    title(sprintf('Page %d / %d. ',page,npages));
  end;
  if page ~= npages,
    waitforbuttonpress;
  end;

end;

