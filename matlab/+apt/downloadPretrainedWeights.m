function downloadPretrainedWeights(varargin) 
  aptroot = myparse(varargin,...
    'aptroot',APT.Root...
    );
  
  urlsAll = DeepTracker.pretrained_weights_urls;
  weightfilepats = DeepTracker.pretrained_weights_files_pat_lnx;
  deepnetrootlnx = [aptroot '/deepnet'];
  pretrainedlnx = [deepnetrootlnx '/pretrained'];
  for i = 1:numel(urlsAll)
    url = urlsAll{i};
    pat = weightfilepats{i};
    wfile = sprintf(pat,deepnetrootlnx);

    if exist(wfile,'file')>0
      fprintf('Tensorflow resnet pretrained weights %s already downloaded.\n',url);
      continue;
    end
      
    % hmm what happens when the weightfilenames change?
    fprintf('Downloading tensorflow resnet pretrained weights %s (APT)..\n',url);
    outfiles = untar(url,pretrainedlnx);
    sprintf('Downloaded and extracted the following files/directories:\n');
    fprintf('%s\n',outfiles{:});
  end      
end

