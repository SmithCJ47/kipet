from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.common.collections import ComponentSet
from pyomo.dae.flatten import flatten_dae_components
from pyomo.util.subsystems import (
    generate_subsystem_blocks,
    TemporarySubsystemManager,
)
from pyomo.core.expr.visitor import identify_variables

from pyomo.contrib.incidence_analysis import IncidenceGraphInterface, dulmage_mendelsohn

import time

def _filter_duplicates(comps):
    seen = set()
    for comp in comps:
        if id(comp) not in seen:
            seen.add(id(comp))
            yield comp
#%%

def get_subsystems_along_time(m, time, include_scalar_cons=True):
    """
    Arguments
    ---------
    m: Block
        A model containing components indexed by time
    time: Set
        A set indexing components of the model

    """
    #%%
    # from pyomo.dae import ContinuousSet
    # m = r1._s_model
    
    # time_index = None
    # for i in m.component_objects(ContinuousSet):
    #     time_index = i
    #     break
    # if time_index is None:
    #     raise Exception('No ContinuousSet')

    # time = time_index
    # include_scalar_cons=True
    
    scalar_vars, dae_vars = flatten_dae_components(m, time, Var)
    scalar_cons, dae_cons = flatten_dae_components(m, time, Constraint)

    
    # First get constraints
    subsystem_cons = [list(_filter_duplicates(
        con[t] for con in dae_cons if t in con and con.active and con[t].active
    )) for t in time
    ]
    if include_scalar_cons:
        # Add scalar constraints; These presumably specify initial conditions
        subsystem_cons[0] += scalar_cons

    # Then get variables in those constraints
    con_vars = dict()
    seen = set()
    vars_in_cons = []
    active_scalar_vars = []
    
    scalar_var_id = [id(var) for var in scalar_vars]
    
    for i, t in enumerate(time):
        con_vars[t] = ComponentSet()
        for con in subsystem_cons[i]:
            for var in identify_variables(con.expr, include_fixed=False):
                con_vars[t].add(var)
                if id(var) not in seen:
                    seen.add(id(var))
                    vars_in_cons.append(var)
                    if id(var) in scalar_var_id:
                        active_scalar_vars.append(id(var))
                    
    # There must be a cleaner way of doing this
    #active_cons = list(m.component_data_objects(Constraint, active=True))
    #con_names = [con.name for con in active_cons]
    #print(f'{len(active_cons) = }')
    #print(f'{len(vars_in_cons) = }')
    
    # vars_in_cons_set = set([s.name for s in vars_in_cons])
    # scalar_vars_set = set([s.name for s in scalar_vars])
    # active_scalar_vars = list(vars_in_cons_set.intersection(scalar_vars_set))
    
    #print(f'{active_scalar_vars = }')

    # Then organize variables by time point
    subsystem_vars = [list(_filter_duplicates(
        var[t] for var in dae_vars if t in var and var[t] in con_vars[t]
    )) for t in time
    ]
    if include_scalar_cons:
        for var in vars_in_cons:
            if id(var) in active_scalar_vars:
                # Add scalar constraints; These presumably specify initial conditions
                subsystem_vars[0] += [var]

    subsystems = [
        (cons, vars) for cons, vars in zip(subsystem_cons, subsystem_vars)
    ]
    
    # import pprint as pt
    # #pt.pprint([s.name for s in scalar_vars])
    # pt.pprint([s.name for s in subsystems[0][0]])
    # pt.pprint([s.name for s in subsystems[0][1]])
    
    # # Missing 
    # pt.pprint([s.name for s in subsystems[127][0]])
    # pt.pprint([s.name for s in subsystems[127][1]])
    
    # pt.pprint([s.name for s in subsystems[128][0]])
    # pt.pprint([s.name for s in subsystems[128][1]])
    
    for i, ss in enumerate(subsystems):
        print(f'{i}: {len(ss[0])} : {len(ss[1])}')
    
    
    #%%
    #subsystems = [
    #    (
    #        list(_filter_duplicates(
    #            con[t] for con in dae_cons
    #            if t in con and con.active and con[t].active
    #        )),
    #        list(_filter_duplicates(
    #            var[t] for var in dae_vars if t in var and var[t] in con_vars[t]
    #        )),
    #    )
    #    for t in time
    #]
    return subsystems

#%%
def initialize_by_time_element(m, time, solver, solve_kwds=None):
    if solve_kwds is None:
        solve_kwds = {}
    reslist = []
    for block, inputs in generate_time_element_blocks(m, time):
        with TemporarySubsystemManager(to_fix=inputs):
            solver.solve(block, **solve_kwds)


def generate_time_element_blocks(m, time):
    scalar_vars, dae_vars = flatten_dae_components(m, time, Var)
    scalar_cons, dae_cons = flatten_dae_components(m, time, Constraint)
    subsystems = get_subsystems_along_time(m, time)
    partition = partition_independent_subsystems(subsystems)
    time_partition = [[time.at(i+1) for i in subset] for subset in partition]

    combined_subsystems = [
        (
            [con for i in subset for con in subsystems[i][0]],
            [var for i in subset for var in subsystems[i][1]],
        )
        for subset in partition
    ]
    for i, (block, inputs) in enumerate(
            generate_subsystem_blocks(combined_subsystems)
            ):
        t_points = time_partition[i]
        assert len(block.vars) == len(block.cons)
        if i != 0:
            # Initialize with results of previous solve
            for t in t_points:
                for var in dae_vars:
                    if not var[t].fixed:
                        var[t].set_value(var[latest].value)
        yield block, inputs
        # I don't think t_points can be empty. TODO: is this correct?
        latest = t_points[-1]


def generate_time_blocks(m, time):
    subsystems = get_subsystems_along_time(m, time)
    for block, inputs in generate_subsystem_blocks(subsystems):
        yield block, inputs


def partition_independent_subsystems(subsystems):
    """
    This method takes a partition of a model into potentially independent
    subsets, and combines these subsets if any of them are mutually
    dependent according to a block triangularization.

    """
    n_subsystem = len(subsystems)
    total_vars = [var for _, variables in subsystems for var in variables]
    total_cons = [con for constraints, _ in subsystems for con in constraints]
    igraph = IncidenceGraphInterface()
    
    v_b_map, c_b_map = igraph.block_triangularize(total_vars, total_cons)
    blocks_per_subsystem = [set() for _ in range(n_subsystem)]
    for i in range(n_subsystem):
        constraints, variables = subsystems[i]
        con_blocks = set(c_b_map[con] for con in constraints)
        var_blocks = set(v_b_map[var] for var in variables)
        # Here we require that the user's subsystems be compatible
        # with a block triangularization. I.e. that the union of diagonal
        # blocks is the same for variables and constraints.
        # Why do we require this again? Because it is required for these
        # time blocks to be square systems that we can solve.
        assert con_blocks == var_blocks
        blocks_per_subsystem[i].update(con_blocks)
    n_blocks = len(set(v_b_map.values()))
    subsystems_per_block = [set() for _ in range(n_blocks)]
    for i in range(n_subsystem):
        for b in blocks_per_subsystem[i]:
            subsystems_per_block[b].add(i)

    # These will partition our subsystems
    subsets = [set((i,)) for i in range(n_subsystem)]
    for subsystems in subsystems_per_block:
        # If a block contains multiple subsystems, these subsystems must
        # be solved simultaneously
        s0 = subsystems.pop()
        for s in subsystems:
            # Combine the subsets for these subsystems
            subsets[s0].update(subsets[s])
            subsets[s] = subsets[s0]
    unique_subsets = list(_filter_duplicates(subsets))
    sorted_subsets = [list(sorted(s)) for s in unique_subsets]
    return sorted_subsets
