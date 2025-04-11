function testBackendConfigUI(backend, cacheDir)
  % Test whether backend is ready to do; display results in msgbox
  
  switch backend.type,
    case DLBackEnd.Bsub,
      testBsubBackendConfig(backend, cacheDir);
    case DLBackEnd.Docker
      testDockerBackendConfig(backend);
    case DLBackEnd.AWS
      testAWSBackendConfig(backend);
    case DLBackEnd.Conda
      testCondaBackendConfig(backend);
    otherwise
      msgbox(sprintf('Tests for %s have not been implemented',backend.type),...
             'Not implemented','modal');
  end
end
