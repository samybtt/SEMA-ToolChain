import logging
import angr

import configparser

config = configparser.ConfigParser()
config.read('config.ini')
lw = logging.getLogger("CustomSimProcedureWindows")
lw.setLevel(config['SCDG_arg'].get('log_level'))


class WinHttpAddRequestHeaders(angr.SimProcedure):
    def run(self, arg1, arg2, arg3):
        return self.state.solver.BVS(
            "retval_{}".format(self.display_name), self.arch.bits
        )
