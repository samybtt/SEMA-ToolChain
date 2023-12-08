import logging
import angr
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
lw = logging.getLogger("CustomSimProcedureWindows")
lw.setLevel(config['SCDG_arg'].get('log_level'))


class AfxWinMain(angr.SimProcedure):
    def run(
        self,
        arg1,
        arg2,
        arg3,
        arg4
    ):
        print(hex(self.state.solver.eval(arg1)))
        print(hex(self.state.solver.eval(arg2)))
        print(hex(self.state.solver.eval(arg3)))
        print(hex(self.state.solver.eval(self.state.memory.load(self.state.solver.eval(arg3),16))))
        print(hex(self.state.solver.eval(arg4)))
        return 0x666
