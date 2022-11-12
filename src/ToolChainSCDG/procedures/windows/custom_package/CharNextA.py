import logging
import angr

lw = logging.getLogger("CustomSimProcedureWindows")

class CharNextA(angr.SimProcedure):
    def run(self, ptr):
        # import pdb; pdb.set_trace()
        return self.state.solver.If(self.state.mem[ptr].uint8_t.resolved == 0, ptr, ptr + 1)