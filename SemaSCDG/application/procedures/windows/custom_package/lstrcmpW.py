import logging
import angr

import configparser

config = configparser.ConfigParser()
config.read('config.ini')
lw = logging.getLogger("CustomSimProcedureWindows")
lw.setLevel(config['SCDG_arg'].get('log_level'))


class lstrcmpW(angr.SimProcedure):
    def run(self, string1, string2):
        if string1.symbolic or string2.symbolic:
            return self.state.solver.BVS("retval_{}".format(self.display_name), 32)
            
        try:
            first_str = self.state.mem[string1].wstring.concrete
        except:
            lw.debug("string1 not resolvable")
            first_str = ""
        try:
            second_str = self.state.mem[string2].wstring.concrete
        except:
            lw.debug("string2 not resolvable")
            second_str = ""
            
        if first_str == second_str:
            return 0
        elif first_str > second_str:
            return 0x1
        else:
            return -1
