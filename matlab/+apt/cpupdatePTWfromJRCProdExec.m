function cpupdatePTWfromJRCProdExec(aptrootLnx) % throws if errors
  cmd = DeepTracker.cpPTWfromJRCProdLnx(aptrootLnx);
  cmd = wrapCommandSSH(cmd,'host',DLBackEndClass.jrchost);
  apt.syscmd(cmd,'failbehavior','err');
end
