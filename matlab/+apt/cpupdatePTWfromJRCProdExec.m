function cpupdatePTWfromJRCProdExec(aptrootLnx) % throws if errors
  cmd = DeepTracker.cpPTWfromJRCProdLnx(aptrootLnx);
  cmd = wrapCommandSSH(cmd,'host',DLBackEndClass.jrchost);
  DeepTracker.syscmd(cmd,'failbehavior','err');
end

