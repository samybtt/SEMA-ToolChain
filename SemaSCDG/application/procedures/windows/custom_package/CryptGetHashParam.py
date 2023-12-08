import logging
import angr
import claripy
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
lw = logging.getLogger("CustomSimProcedureWindows")
lw.setLevel(config['SCDG_arg'].get('log_level'))


class CryptGetHashParam(angr.SimProcedure):
    def run(
        self,
        hHash,
        dwParam,
        pbData,
        pdwDataLen,
        dwFlags
    ):
        ptr1 = claripy.BVV(0x10, self.arch.bits)
        self.state.memory.store(pdwDataLen,ptr1)
        ptr = claripy.BVV(self.state.globals["crypt_result"])
        self.state.memory.store(pbData,ptr)
        return 0x1
