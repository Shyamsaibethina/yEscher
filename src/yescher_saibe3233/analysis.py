import code2.io as io
from escher import Builder
from cobra.util.solver import set_objective
from pytfa.optim.utils import symbol_sum
from etflMain.etfl.optim.constraints import ModelConstraint
from etflMain.etfl.analysis.dynamic import compute_center
from etflMain.etfl.optim.utils import fix_growth, release_growth, safe_optim
from time import time
import pandas as pd
import numpy as np
import importlib
import etflMain
from etflMain.etfl.io.json import load_json_model
import sys

importlib.reload(etflMain.etfl.io)

flux_to_set = 'growth'
# flux_to_set = 'glucose'

solver = 'optlang-gurobi'
# solver = 'optlang-cplex'

GLC_RXN_ID = 'r_1714'
GROWTH_RXN_ID = 'r_4041'

constrainedUptake = ['r_1604', 'r_1639', 'r_1873', 'r_1879', 'r_1880',
                     'r_1881', 'r_1671', 'r_1883', 'r_1757', 'r_1891', 'r_1889', 'r_1810',
                     'r_1993', 'r_1893', 'r_1897', 'r_1947', 'r_1899', 'r_1900', 'r_1902',
                     'r_1967',
                     # 'r_1918', Y7 doesn't have a linoleic acid exchange rxn, which
                     # was substituted for octadecenoate and octadecynoate for Y5 and Y6,
                     # so leave out.
                     'r_1903', 'r_1548', 'r_1904', 'r_2028',
                     'r_2038', 'r_1906', 'r_2067', 'r_1911', 'r_1912', 'r_1913', 'r_2090',
                     'r_1914', 'r_2106']   # potassium exchange

unconstrainedUptake = ['r_1714', 'r_1672', 'r_1654',  # ammonium exchange
                       'r_1992',  # oxygen exchange
                       'r_2005',  # phosphate exchange
                       'r_2060',  # sulphate exchange
                       'r_1861',  # iron exchange, for test of expanded biomass def
                       'r_1832',  # hydrogen exchange
                       'r_2100',  # water exchange
                       'r_4593',  # chloride exchange
                       'r_4595',  # Mn(2+) exchange
                       'r_4596',  # Zn(2+) exchange
                       'r_4597',  # Mg(2+) exchange
                       'r_2049',  # sodium exchange
                       'r_4594',  # Cu(2+) exchange
                       'r_4600',  # Ca(2+) exchange
                       'r_2020']

anyUptake = constrainedUptake + unconstrainedUptake  # for objective function

uptake_range = pd.Series(np.arange(-0.5, -15.5, -0.5))


def _chemostat_sim(model):
    # growth media should not be changed, it's minimal mineral

    # applying gecko's rxn modifications in order:
    model.reactions.r_1549.upper_bound = 1e-5  # butanediol secretion
    model.reactions.r_2033.upper_bound = 0.05  # pyruvate secretion
    model.reactions.r_1631.upper_bound = 1e-5  # acetaldehyde secretion
    model.reactions.r_1810.upper_bound = 1e-5  # glycine secretion
    model.reactions.r_1634.upper_bound = 0.62  # acetate secretion

    # rxn block (gecko)
    model.reactions.r_0659.lower_bound = 0  # isocitrate dehydrogenase (NADP)
    model.reactions.r_0659.upper_bound = 0
    model.reactions.r_2045.lower_bound = 0  # L-serine transport

    return


# def _va_sim(model):
#     model.objective.direction = 'max'
#     sol_max = safe_optim(model)

#     model.objective.direction = 'min'
#     sol_min = safe_optim(model)

#     return sol_min, sol_max


def _prep_sol(substrate_uptake, model):

    ret = {'obj': model.solution.objective_value,
           'mu': model.solution.fluxes.loc[model.growth_reaction.id],
           'available_substrate': -1*substrate_uptake,
           'uptake': -1*model.solution.fluxes[GLC_RXN_ID]
           }

#    for exch in model.exchanges:
#        ret[exch.id] = model.solution.fluxes.loc[exch.id]
    for rxn in model.reactions:
        ret[rxn.id] = model.solution.fluxes.loc[rxn.id]
    for enz in model.enzymes:
        ret['EZ_' + enz.id] = model.solution.raw.loc['EZ_'+enz.id]

    return pd.Series(ret)


ctrl_model = load_json_model(
    "src/yescher_saibe3233/input_model/yeast8_cEFL_2584_enz_128_bins__20240209_125642.json")
cobra_model = io.read_yeast_model()


def knockout(growth_rate=0, knockouts=[], map_file_path="", csv_file_path="", map_name=""):
    for knockout in knockouts:
        # Load the model
        model = ctrl_model.copy()

        # Knockout the gene
        the_trans = model.get_translation(knockout)
        the_trans.upper_bound = 0

        data = {}
        sol = pd.Series()
        # chebyshev_variables = None
        # BIGM = 1000

        model.warm_start = None
        model.logger.info('Simulating ...')
        start = time()

        tol = 0.01
        _chemostat_sim(model)
        model.reactions.get_by_id(GLC_RXN_ID).upper_bound = 0
        model.reactions.get_by_id(GLC_RXN_ID).lower_bound = -1000
        # minimize substrate uptake
        model.objective = symbol_sum([model.reactions.get_by_id(x).reverse_variable
                                      for x in anyUptake])
        model.objective_direction = 'min'

        model.reactions.get_by_id(GROWTH_RXN_ID).upper_bound = growth_rate
        model.reactions.get_by_id(GROWTH_RXN_ID).lower_bound = growth_rate

        temp_sol = safe_optim(model)
        upt = model.objective.value
        expr = model.objective.expression
        sub_cons = model.add_constraint(kind=ModelConstraint,
                                        hook=model,
                                        expr=expr,
                                        id_='fix_substrate',
                                        lb=upt - abs(tol * upt),
                                        ub=upt + abs(tol * upt),)
        # fix growth
        fix_growth(model)
        # minimize total sum of fluxes
        model.objective = symbol_sum([model.reactions.get_by_id(x.id).forward_variable +
                                      model.reactions.get_by_id(
            x.id).reverse_variable
            for x in ctrl_model.reactions
            if x.id != 'r_4050'])  # only metabolic reactions and exclude dna reaction as it's not in vETFL
        print(type(model.objective))
        model.slim_optimize()

        # fix sum of fluxes
        rhs = model.objective.value
        expr = model.objective.expression
        flux_cons = model.add_constraint(kind=ModelConstraint,
                                         hook=model,
                                         expr=expr,
                                         id_='fix_tot_flux',
                                         lb=rhs - abs(tol * rhs),
                                         ub=rhs + abs(tol * rhs),)
        # minimize enzyme usage i.e. max dummy enzyme
        obj_expr = symbol_sum([model.enzymes.dummy_enzyme.variable])
        set_objective(model, obj_expr)
        model.objective_direction = 'max'

        model.optimize()
        # this also fixes growth with fix_growth function
        chebyshev_sol = compute_center(
            model, model.objective, provided_solution=model.solution)
        #                 chebyshev_sol = compute_center(
        #                     model, model.solution)
        new_sol = _prep_sol(upt, model)
        sol = pd.concat([sol, new_sol], axis=1)
        # revert the changes
        release_growth(model)
        model.remove_constraint(sub_cons)
        model.remove_constraint(flux_cons)

        data[knockout] = sol
        stop = time()
        print('Elapsed time: {}'.format(stop - start))
        data[knockout].to_csv(
            csv_file_path + "{}_knockout.csv".format(knockout))

        df = pd.read_csv(csv_file_path +
                         "{}_knockout.csv".format(knockout))

        # Get rid of the first 4 rows
        df = df[4:]

        # Get rid of the second column
        df = df.drop(df.columns[1], axis=1)

        # Remove the rows that do not start with r_ in the first column
        df = df[df[df.columns[0]].str.startswith('r_')]
        df = df.reset_index(drop=True)

        flux_dictionary_name = {}
        for rxn in cobra_model.reactions:
            try:
                # Finds the value in df that corresponds to the reaction id in the model
                flux = df.loc[df[df.columns[0]] == rxn.id].iloc[0][1]
                # print(flux)

                flux_dictionary_name[rxn.annotation['bigg.reaction']] = flux
            except:
                pass

        map = Builder(map_name=map_name)
        map.reaction_data = flux_dictionary_name
        map.save_html(
            map_file_path + "{}_knockout_map.html".format(knockout))


# YLR044C is the gene for the translation of ribosomal proteins
knockout(0.40, ['YLR044C'], "src/yescher_saibe3233/outputs/",
         "src/yescher_saibe3233/outputs/", "iMM904.Central carbon metabolism")
