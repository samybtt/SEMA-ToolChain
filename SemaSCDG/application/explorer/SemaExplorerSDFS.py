#!/usr/bin/env python3
import monkeyhex  # this will format numerical results in hexadecimal
import logging
import sys
import angr
from SemaExplorer import SemaExplorer

class SemaExplorerSDFS(SemaExplorer):
    def __init__(
        self,
        simgr,
        exp_dir,
        nameFileShort,
        scdg_graph,
        call_sim,
        proj,
        find = 0
    ):
        super(SemaExplorerSDFS, self).__init__(
            simgr,
            exp_dir,
            nameFileShort,
            scdg_graph,
            call_sim
        )
        self.proj = proj
        self.find = find
        self.log = logging.getLogger("SemaExplorerSDFS")
        self.log.setLevel("INFO")
        
    def execute_concretely(self, simgr, proj, sdfs):
        if len(simgr.stashes["ExcessLoop"]) > 0:
            state = simgr.stashes["ExcessLoop"][0]
            irsb = proj.factory.block(state.history.jump_source).vex
            find_addr = 0
            jump = list(irsb.constant_jump_targets)
            target = state.solver.eval(state.history.jump_target)
            source = state.solver.eval(state.history.jump_source)
            print("1 " + hex(target))
            print("2 " + hex(source))
            if(target == jump[0]):
                find_addr = jump[1]
            else:
                find_addr = jump[0]
            print("3 " + hex(find_addr))
            simgr.move(
                from_stash="ExcessLoop",
                to_stash="active",
                filter_func=lambda s: True,
            )
            simgr.remove_technique(sdfs)
            tech = angr.exploration_techniques.Symbion(find=[find_addr])
            simgr.use_technique(tech)
            simgr.run()
            simgr.move(
                from_stash="found",
                to_stash="active",
                filter_func=lambda s: True,
            )
            simgr.remove_technique(tech)
            simgr.use_technique(sdfs)
            exploration = simgr.run()
            
    def step(self, simgr, stash="active", **kwargs):
        try:
            simgr = simgr.step(stash=stash, **kwargs)
        except Exception as inst:
            self.log.warning("ERROR IN STEP() - YOU ARE NOT SUPPOSED TO BE THERE !")
            # self.log.warning(type(inst))    # the exception instance
            self.log.warning(inst)  # __str__ allows args to be printed directly,
            exc_type, exc_obj, exc_tb = sys.exc_info()
            # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            self.log.warning(exc_type, exc_obj)
            exit(-1)
        super().build_snapshot(simgr)

        if self.verbose and (
            len(self.fork_stack) > 0 or len(simgr.deadended) > self.deadended
        ):
            self.log.info(
                "A new block of execution have been executed with changes in sim_manager."
            )
            self.log.info("Currently, simulation manager is :\n" + str(simgr))
            self.log.info("pause stash len :" + str(len(simgr.pause)))

        if self.verbose and len(self.fork_stack) > 0:
            self.log.info("fork_stack : " + str(len(self.fork_stack)) + " " + hex(simgr.active[0].addr) + " " + hex(simgr.active[1].addr))
        
        simgr.move("active", "found", lambda s: s.addr == self.find)

        # We detect fork for a state
        super().manage_fork(simgr)
        
        #super().execute_concretely(simgr, self.proj, self)
            
        # If limit of simultaneous state is not reached and we have some states available in pause stash
        if len(simgr.stashes["pause"]) > 0 and len(simgr.active) < self.max_simul_state:
            moves = min(
                self.max_simul_state - len(simgr.active),
                len(simgr.stashes["pause"]),
            )
            for m in range(moves):
                super().take_longuest(simgr, "pause")
                    

        super().manage_pause(simgr)
        
        super().drop_excessed_loop(simgr)

        # If states end with errors, it is often worth investigating. Set DEBUG_ERROR to live debug
        # TODO : add a log file if debug error is not activated
        super().manage_error(simgr)

        super().manage_unconstrained(simgr)

        for vis in simgr.active:
            self.dict_addr_vis.add(str(super().check_constraint(vis, vis.history.jump_target)))
            
        super().excessed_step_to_active(simgr)

        super().excessed_loop_to_active(simgr)

        super().time_evaluation(simgr)
        
        return simgr
