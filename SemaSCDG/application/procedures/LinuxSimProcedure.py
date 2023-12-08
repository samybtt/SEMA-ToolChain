import logging

from clogging.CustomFormatter import CustomFormatter
from CustomSimProcedure import CustomSimProcedure
from angr.calling_conventions import  SimCCSystemVAMD64

logger = logging.getLogger("LinuxSimProcedure")
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)
logger.propagate = False
logger.setLevel(logging.INFO)

class LinuxSimProcedure(CustomSimProcedure):

    def __init__(self, log_level):
        super().__init__()
        self.log_level = log_level
        self.init_sim_proc("linux")
        self.config_logger()

    def config_logger(self):
        self.log = logging.getLogger("LinuxSimProcedure")
        ch = logging.StreamHandler()
        ch.setLevel(self.log_level)
        ch.setFormatter(CustomFormatter())
        self.log.addHandler(ch)
        self.log.propagate = False
        self.log.setLevel(self.log_level)

    def deal_with_alt_names(self, pkg_name, proc):
        for altname in proc.ALT_NAMES:
            self.sim_proc[pkg_name][altname] = proc
    
    def custom_hook_linux_symbols(self, proj):
        """_summary_
        TODO CH
        Args:
            proj (_type_): _description_
        """
        # self.ANG_CALLING_CONVENTION = {"__stdcall": SimCCStdcall, "__cdecl": SimCCCdecl}
        self.log.info("custom_hook_linux_symbols")
        proj.loader
        symbols = proj.loader.symbols

        for symb in symbols:
            if symb.name in self.sim_proc["custom_package"]:
                # if "CreateThread" in symb.name:
                #     self.create_thread.add(symb.rebased_addr)
                proj.unhook(symb.rebased_addr)
                if not self.amd64_sim_proc_hook(proj, symb.rebased_addr, self.sim_proc["custom_package"][symb.name]):
                    if symb.name not in self.CDECL_EXCEPT:
                        self.std_sim_proc_hook(proj, symb.rebased_addr, self.sim_proc["custom_package"][symb.name])
                    else:
                        self.exception_sim_proc_hook(proj, symb.rebased_addr, self.sim_proc["custom_package"][symb.name])


    def amd64_sim_proc_hook(self, project, name, sim_proc):
        if project.arch.name == "AMD64":
            project.hook(
                name,
                sim_proc(
                    cc=SimCCSystemVAMD64(project.arch)
                ),
            )
            return True
        return False
            