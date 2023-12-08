import logging
import sys
import angr
import archinfo

import configparser

config = configparser.ConfigParser()
config.read('config.ini')
lw = logging.getLogger("CustomSimProcedureWindows")
lw.setLevel(config['SCDG_arg'].get('log_level'))

class LockResource(angr.SimProcedure):
    def run(self, hResData):
        if self.state.solver.eval(hResData) == 0x0 or hResData.symbolic:
            ptr=self.state.solver.BVS("buf",8*0x20,key=("buffer", hex(self.state.globals["n_buffer"])),eternal=True)
            addr = self.state.heap._malloc(0x20)
            self.state.memory.store(addr,ptr,endness=archinfo.Endness.LE)
            return addr
        return hResData
           
        
