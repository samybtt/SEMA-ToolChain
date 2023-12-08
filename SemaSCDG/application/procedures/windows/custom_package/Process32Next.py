import logging
import angr

import configparser

config = configparser.ConfigParser()
config.read('config.ini')
lw = logging.getLogger("CustomSimProcedureWindows")
lw.setLevel(config['SCDG_arg'].get('log_level'))


class Process32Next(angr.SimProcedure):
    def run(self, hSnapshot, lppe):
        return 0x0
    
    # TODO list of usual process
