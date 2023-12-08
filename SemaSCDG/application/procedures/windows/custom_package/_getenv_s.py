import angr
import logging

import configparser

config = configparser.ConfigParser()
config.read('config.ini')
lw = logging.getLogger("CustomSimProcedureWindows")
lw.setLevel(config['SCDG_arg'].get('log_level'))

class _getenv_s(angr.SimProcedure):
    def get_str(self, lpName):
        name = self.state.mem[lpName].string.concrete
        if hasattr(name, "decode"):
            name = name.decode("utf-8")
        name = name.upper()
        lw.debug(name)
        if name in self.state.plugin_env_var.env_var.keys() and self.state.plugin_env_var.env_var[name] != None:
            ret = self.state.plugin_env_var.env_var[name]
            lw.debug(ret)
            # lw.warning(name + " " + str(size) + " " + ret)
            try:  # TODO investigate why needed with explorer
                if ret[-1] != "\0":
                    ret += "\0"
            except IndexError:
                lw.warning("IndexError - GetEnvironmentVariableA")
                ret = "\0"
            if hasattr(ret, "encode"):
                ret = ret.encode("utf-")
        else:
            ret =  None #"None"
            self.state.plugin_env_var.env_var[name] = None
        lw.debug(ret)
        self.state.plugin_env_var.env_var_requested[name] = ret
        return ret


    def run(self, pReturnValue, buffer, numberOfElements, varname):
        if varname.symbolic:
            return self.state.solver.BVS(
                "retval_{}".format(self.display_name), self.arch.bits
            )
        # Get the environment variable value
        env_value = self.get_str(varname)

        # If the environment variable is not defined, set the return value to EINVAL
        if env_value is None:
            return self.state.solver.BVS(
                    "retval_{}".format(self.display_name), self.arch.bits
            )

        # If the environment variable is defined, copy its value to the buffer
        buf_size = min(len(env_value), self.state.solver.eval(numberOfElements) - 1)
        self.state.memory.store(buffer, env_value[:buf_size] + b"\0")

        # Set the return value to zero and update pReturnValue with the number of characters copied
        self.state.memory.store(pReturnValue, buf_size, self.state.arch.bits)
        return 0x0 #self.state.solver.BVV(0, self.state.arch.bits)
