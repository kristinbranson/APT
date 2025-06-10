function downloadPretrainedWeights(varargin) 
  % aptroot is a native path
  aptroot = myparse(varargin,...
    'aptroot',APT.Root...
    );
  
  urlsAll = DeepTracker.pretrained_weights_urls;
  weightfilerelpaths = DeepTracker.pretrained_weights_files_relative_path;
  deepnetrootnative = fullfile(aptroot, 'deepnet');  % native path
  pretrainednative = fullfile(deepnetrootnative, 'pretrained') ;
  for i = 1:numel(urlsAll)
    url = urlsAll{i};
    weightfilerelpath = weightfilerelpaths{i};
    weightfilepath = fullfile(deepnetrootnative, weightfilerelpath) ;  % native path

    if exist(weightfilepath,'file')
      fprintf('Tensorflow resnet pretrained weights %s already downloaded.\n',url);
      continue
    end
      
    % hmm what happens when the weightfilenames change?
    fprintf('Downloading tensorflow resnet pretrained weights %s (APT)...\n',url);
    outfiles = untar(url,pretrainednative);
    sprintf('Downloaded and extracted the following files/directories:\n');
    fprintf('%s\n',outfiles{:});
  end      
end
