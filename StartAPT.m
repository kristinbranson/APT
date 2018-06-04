function apt = StartAPT
if APT.pathNotConfigured
  fprintf('Configuring your path ...\n');
  APT.setpath;
end
apt = Labeler();