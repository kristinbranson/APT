function cfg = readDefaultCfg()
  % Read the default project config from the default config YAML file.
  %
  % In the YAML file the optional fields ViewNames and LabelPointNames are left
  % blank, and yaml.ReadYaml() represents a blank field as an empty double
  % array ([]) rather than an empty cell array ({}).  Downstream code expects
  % these fields to be cell arrays, so if either is empty and numeric, replace
  % it with cell(1,0).
  DEFAULT_CFG_FILENAME = fullfile(APT.Root, 'matlab', 'config.default.yaml') ;
  cfg = yaml.ReadYaml(DEFAULT_CFG_FILENAME) ;
  if isempty(cfg.ViewNames) && isnumeric(cfg.ViewNames)
    cfg.ViewNames = cell(1,0) ;
  end
  if isempty(cfg.LabelPointNames) && isnumeric(cfg.LabelPointNames)
    cfg.LabelPointNames = cell(1,0) ;
  end
end  % function
