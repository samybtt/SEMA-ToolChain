import angr
import logging

from .VirtualAlloc import convert_prot, deconvert_prot

import os

lw = logging.getLogger("CustomSimProcedureWindows")
lw.setLevel(os.environ["LOG_LEVEL"])

class VirtualProtect(angr.SimProcedure):
    def run(self, lpAddress, dwSize, flNewProtect, lpfOldProtect):
        return 1
