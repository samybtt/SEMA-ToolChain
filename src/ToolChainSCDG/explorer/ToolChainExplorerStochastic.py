import logging
import random
from collections import defaultdict, deque
import sys

# from angr.exploration_techniques import ExplorationTechnique
from .ToolChainExplorer import ToolChainExplorer

l = logging.getLogger('syml')


class ToolChainExplorerStochastic(ToolChainExplorer):
    """
    Stochastic Search.
    Will only keep one path active at a time, any others will be discarded.
    Before each pass through, weights are randomly assigned to each basic block.
    These weights form a probability distribution for determining which state remains after splits.
    When we run out of active paths to step, we start again from the start state.
    """

    def __init__(
        self,
        simgr,
        max_length,
        exp_dir,
        nameFileShort,
        worker,
        restart_prob=0.0001
    ):
        """
        :param start_state:  The initial state from which exploration stems.
        :param restart_prob: The probability of randomly restarting the search (default 0.0001).
        """
        super(ToolChainExplorerStochastic, self).__init__(
            simgr,
            max_length,
            exp_dir,
            nameFileShort,
            worker
        )
        self.start_state = simgr.one_active
        self.restart_prob = restart_prob
        self._random = random.Random()
        self._random.seed(42)
        self.affinity = defaultdict(self._random.random)
        # self.pause_stash = deque()
        self.log = logging.getLogger("ToolChainExplorerStochastic")
        self.log.setLevel("INFO")

    def step(self, simgr, stash='active', **kwargs):
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

        if self.print_sm_step and (
            len(self.fork_stack) > 0 or len(simgr.deadended) > self.deadended
        ):
            self.log.info(
                "A new block of execution have been executed with changes in sim_manager.\n"
            )
            self.log.info("Currently, simulation manager is :\n" + str(simgr))
            self.log.info("pause stash len :" + str(len(self.pause_stash)))

        if self.print_sm_step and len(self.fork_stack) > 0:
            self.log.info("fork_stack : " + str(len(self.fork_stack)))

        # We detect fork for a state
        super().manage_fork(simgr)

        # Remove state which performed more jump than the limit allowed
        super().remove_exceeded_jump(simgr)

        # Manage ended state
        super().manage_deadended(simgr)

        super().mv_bad_active(simgr)
        def weighted_pick(states):
                """
                param states: Diverging states.
                """
                # import pdb; pdb.set_trace()
                assert len(states) >= 2
                total_weight = sum((self.affinity[s.addr] for s in states))
                selected = self._random.uniform(0, total_weight)
                i = 0
                
                for i, state in enumerate(states):
                    weight = self.affinity[state.addr]
                    if selected < weight:
                        break
                    else:
                        selected -= weight
                picked = states[i]
                return picked

        if self._random.random() < self.restart_prob:
            simgr.active = [self.start_state]
            self.affinity.clear()
        elif not simgr.active:
            if len(simgr.stashes["pause"]) > 0:
                moves = min(
                self.max_simul_state,
                len(simgr.stashes["pause"]),
                )
                for m in range(moves):
                    simgr.move(
                        from_stash="pause",
                        to_stash="active",
                        filter_func=lambda s: s.addr == simgr.stashes["pause"][m].addr
                    )
            else:
                simgr.active = [self.start_state]
                self.affinity.clear()


        if len(simgr.active) > 1:
            # import pdb; pdb.set_trace()
            picked = weighted_pick(simgr.stashes[stash])
            simgr.move(
                from_stash="active",
                to_stash="pause",
                filter_func=lambda s: s != picked #and s not in simgr.stashes["pause"],
            )
            # import pdb; pdb.set_trace()
        # If limit of simultaneous state is not reached and we have some states available in pause stash
        if len(simgr.stashes["pause"]) > 0 and len(simgr.active) < self.max_simul_state:
            moves = min(
                self.max_simul_state - len(simgr.active),
                len(simgr.stashes["pause"]),
            )
            for m in range(moves):
                # import pdb; pdb.set_trace()
                if len(simgr.stashes["pause"]) > 1:
                    picked_from_pause = weighted_pick(simgr.stashes["pause"])
                else:
                    picked_from_pause = simgr.stashes["pause"][0]
                simgr.move(
                    from_stash="pause",
                    to_stash="active",
                    filter_func=lambda s: s == picked_from_pause
                )

        super().manage_pause(simgr)

        super().drop_excessed_loop(simgr)

        # If states end with errors, it is often worth investigating. Set DEBUG_ERROR to live debug
        # TODO : add a log file if debug error is not activated
        super().manage_error(simgr)

        super().manage_unconstrained(simgr)

        for vis in simgr.active:
            self.dict_addr_vis[
                str(super().check_constraint(vis, vis.history.jump_target))
            ] = 1

        super().excessed_step_to_active(simgr)

        super().excessed_loop_to_active(simgr)

        super().time_evaluation(simgr)
        
        return simgr