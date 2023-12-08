import angr
import logging

import configparser

config = configparser.ConfigParser()
config.read('config.ini')
lw = logging.getLogger("CustomSimProcedureWindows")
lw.setLevel(config['SCDG_arg'].get('log_level'))

class RaiseException(angr.SimProcedure):
    # Defining a function called "NO_RET" that does not return anything.
    NO_RET = True
    def run(self, hKey, lpValueName, lpReserved, lpType, lpData, lpcbData):
        # Implement the logic for querying the value of a registry key using the provided parameters.
        lw.warning("RaiseException called")
        return 