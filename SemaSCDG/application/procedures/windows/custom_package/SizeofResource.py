import logging
import sys
import angr
import archinfo

import configparser

config = configparser.ConfigParser()
config.read('config.ini')
lw = logging.getLogger("CustomSimProcedureWindows")
lw.setLevel(config['SCDG_arg'].get('log_level'))

class SizeofResource(angr.SimProcedure):
    def run(self, hModule, hResInfo):
        if self.state.solver.eval(hResInfo) in self.state.plugin_resources.resources:
            lw.debug(hex(self.state.plugin_resources.resources[self.state.solver.eval(hResInfo)]["size"]))
            return self.state.plugin_resources.resources[self.state.solver.eval(hResInfo)]["size"]
        else:
            return 0x20 
            # self.state.solver.BVS(
            #         "retval_{}".format(self.display_name), self.arch.bits
            #     )
           
        
