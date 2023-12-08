import logging
import time as timer
import angr

import configparser

config = configparser.ConfigParser()
config.read('config.ini')
lw = logging.getLogger("CustomSimProcedureWindows")
lw.setLevel(config['SCDG_arg'].get('log_level'))


class GetTickCount(angr.SimProcedure):
    def run(self):
        if angr.options.USE_SYSTEM_TIMES in self.state.options:
            return int(timer.perf_counter() * 1000) + 12345
        else:
            val = self.state.solver.Unconstrained(
                "retval_{}".format(self.display_name), self.arch.bits
            )
            return val
