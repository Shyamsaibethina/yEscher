import os
os.environ['OPTLANG_USE_SYMENGINE'] = "false"

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yEscher.yeastModel.io as io
from escher import Builder
from cobra.util.solver import set_objective
from pytfa.optim.utils import symbol_sum
from yEscher.etfl.etfl.optim.constraints import ModelConstraint
from yEscher.etfl.etfl.analysis.dynamic import compute_center
from yEscher.etfl.etfl.optim.utils import fix_growth, release_growth, safe_optim
from time import time
import pandas as pd
from yEscher.etfl.etfl.io.json import load_json_model
from cobra import Reaction
from typing import List, Dict


# Constants
GLC_RXN_ID = 'r_1714'
GROWTH_RXN_ID = 'r_4041'


# Load models
def load_models(etfl_model_path, cobra_model_path=None):
    """
    Load and return the eTFLa and COBRA models for simulation.

    :param etfl_model_path: Path to the eTFLa model (in JSON format).
    :param cobra_model_path: Optional path to the COBRA model (default is None, which loads the default COBRA model).
    :return: Tuple containing the eTFLa control model and the COBRA model.
    """

    ctrl_model = load_json_model(etfl_model_path)
    cobra_model = io.read_yeast_model() if cobra_model_path is None else io.read_yeast_model(cobra_model_path)
    return ctrl_model, cobra_model

# Uptake reactions
def get_uptake_reactions():
    """
    Retrieve constrained and unconstrained substrate uptake reactions.

    :return: Tuple of lists containing IDs of constrained and unconstrained uptake reactions.
    """

    constrained_uptake = [
        'r_1604', 'r_1639', 'r_1873', 'r_1879', 'r_1880',
        'r_1881', 'r_1671', 'r_1883', 'r_1757', 'r_1891', 'r_1889', 'r_1810',
        'r_1993', 'r_1893', 'r_1897', 'r_1947', 'r_1899', 'r_1900', 'r_1902',
        'r_1967', 'r_1903', 'r_1548', 'r_1904', 'r_2028',
        'r_2038', 'r_1906', 'r_2067', 'r_1911', 'r_1912', 'r_1913', 'r_2090',
        'r_1914', 'r_2106'
    ]
    unconstrained_uptake = [
        'r_1714', 'r_1672', 'r_1654', 'r_1992', 'r_2005', 'r_2060',
        'r_1861', 'r_1832', 'r_2100', 'r_4593', 'r_4595', 'r_4596',
        'r_4597', 'r_2049', 'r_4594', 'r_4600', 'r_2020'
    ]
    return constrained_uptake, unconstrained_uptake

def _chemostat_sim(model):
    """
    Apply modifications to the model's reactions for simulating a chemostat environment.

    :param model: The metabolic model to modify.
    :return: None.
    """

    rxn_modifications = {
        'r_1549': (None, 1e-5),  # butanediol secretion
        'r_2033': (None, 0.05),  # pyruvate secretion
        'r_1631': (None, 1e-5),  # acetaldehyde secretion
        'r_1810': (None, 1e-5),  # glycine secretion
        'r_1634': (None, 0.62),  # acetate secretion
        'r_0659': (0, 0),        # isocitrate dehydrogenase (NADP)
        'r_2045': (0, None)      # L-serine transport
    }
    
    for rxn_id, (lb, ub) in rxn_modifications.items():
        rxn = model.reactions.get_by_id(rxn_id)
        if lb is not None:
            rxn.lower_bound = lb
        if ub is not None:
            rxn.upper_bound = ub

def _prep_sol(substrate_uptake, model):
    """
    Prepare solution results for flux distribution and enzyme analysis.

    :param substrate_uptake: The substrate uptake rate.
    :param model: The metabolic model.
    :return: A pandas Series object containing the fluxes and enzyme activity data.
    """

    ret = {
        'obj': model.solution.objective_value,
        'mu': model.solution.fluxes.loc[model.growth_reaction.id],
        'available_substrate': -1*substrate_uptake,
        'uptake': -1*model.solution.fluxes[GLC_RXN_ID]
    }
    
    for rxn in model.reactions:
        ret[rxn.id] = model.solution.fluxes.loc[rxn.id]
    for enz in model.enzymes:
        ret['EZ_' + enz.id] = model.solution.raw.loc['EZ_'+enz.id]
    
    return pd.Series(ret)

def knockout(growth_rate, knockouts, map_file_path, csv_file_path, map_name, etfl_model_path, cobra_model_path=None):
    """
    Perform gene knockouts and simulate the metabolic model.

    :param growth_rate: The fixed growth rate for the simulation.
    :param knockouts: List of gene knockouts to apply.
    :param map_file_path: Path to save the map output.
    :param csv_file_path: Path to save CSV output of flux results.
    :param map_name: Name of the metabolic map for visualization.
    :param etfl_model_path: Path to the eTFLa model (in JSON format).
    :param cobra_model_path: Optional path to the COBRA model (default is None).
    :return: None.
    """
    
    ctrl_model, cobra_model = load_models(etfl_model_path, cobra_model_path)
    constrained_uptake, unconstrained_uptake = get_uptake_reactions()
    any_uptake = constrained_uptake + unconstrained_uptake

    for knockout in knockouts:
        model = ctrl_model.copy()
        the_trans = model.get_translation(knockout)
        the_trans.upper_bound = 0

        data = {}
        sol = pd.Series()

        model.warm_start = None
        model.logger.info('Simulating ...')
        start = time()

        tol = 0.01
        _chemostat_sim(model)
        model.reactions.get_by_id(GLC_RXN_ID).bounds = (-1000, 0)
        
        # Minimize substrate uptake
        model.objective = symbol_sum([model.reactions.get_by_id(x).reverse_variable for x in any_uptake])
        model.objective_direction = 'min'

        model.reactions.get_by_id(GROWTH_RXN_ID).bounds = (growth_rate, growth_rate)

        temp_sol = safe_optim(model)
        upt = model.objective.value
        expr = model.objective.expression
        sub_cons = model.add_constraint(kind=ModelConstraint, hook=model, expr=expr, id_='fix_substrate',
                                        lb=upt - abs(tol * upt), ub=upt + abs(tol * upt))
        
        fix_growth(model)
        
        # Minimize total sum of fluxes
        model.objective = symbol_sum([model.reactions.get_by_id(x.id).forward_variable +
                                      model.reactions.get_by_id(x.id).reverse_variable
                                      for x in ctrl_model.reactions if x.id != 'r_4050'])
        model.slim_optimize()

        # Fix sum of fluxes
        rhs = model.objective.value
        expr = model.objective.expression
        flux_cons = model.add_constraint(kind=ModelConstraint, hook=model, expr=expr, id_='fix_tot_flux',
                                         lb=rhs - abs(tol * rhs), ub=rhs + abs(tol * rhs))
        
        # Minimize enzyme usage (max dummy enzyme)
        obj_expr = symbol_sum([model.enzymes.dummy_enzyme.variable])
        set_objective(model, obj_expr)
        model.objective_direction = 'max'

        model.optimize()
        chebyshev_sol = compute_center(model, model.objective, provided_solution=model.solution)
        new_sol = _prep_sol(upt, model)
        sol = pd.concat([sol, new_sol], axis=1)
        
        release_growth(model)
        model.remove_constraint(sub_cons)
        model.remove_constraint(flux_cons)

        data[knockout] = sol
        stop = time()
        print(f'Elapsed time: {stop - start}')
        
        # Save and process results
        save_and_process_results(data[knockout], knockout, csv_file_path, map_file_path, map_name, cobra_model)

def save_and_process_results(sol, knockout, csv_file_path, map_file_path, map_name, cobra_model):
    """
    Save the results of the knockout simulation and generate a metabolic map visualization.

    :param sol: Pandas Series containing the flux results of the simulation.
    :param knockout: The gene knockout applied in this simulation.
    :param csv_file_path: Directory path to save CSV file containing flux data.
    :param map_file_path: Directory path to save the metabolic map.
    :param map_name: Name of the metabolic map.
    :param cobra_model: COBRA model used in the simulation for annotation.
    :return: None.
    """
    # Check and create output directories if they do not exist
    os.makedirs(csv_file_path, exist_ok=True)
    os.makedirs(map_file_path, exist_ok=True)

    sol.to_csv(f"{csv_file_path}/{knockout}_knockout.csv")
    
    df = pd.read_csv(f"{csv_file_path}/{knockout}_knockout.csv", skiprows=4)
    df = df.drop(df.columns[1], axis=1)
    df = df[df[df.columns[0]].str.startswith('r_')].reset_index(drop=True)

    flux_dictionary_name = {}
    for rxn in cobra_model.reactions:
        try:
            flux = df.loc[df[df.columns[0]] == rxn.id].iloc[0][1]
            flux_dictionary_name[rxn.annotation['bigg.reaction']] = flux
        except:
            pass

    map = Builder(map_name=map_name)
    map.reaction_data = flux_dictionary_name
    map.save_html(f"{map_file_path}/{knockout}_knockout_map.html")

def perform_flux_variability_analysis(model, reactions: List[str] = None, fraction_of_optimum: float = 1.0):
    """
    Perform Flux Variability Analysis (FVA) on the model.
    
    :param model: The metabolic model
    :param reactions: List of reaction IDs to analyze (default: all reactions)
    :param fraction_of_optimum: The fraction of the optimum objective to consider
    :return: DataFrame with FVA results
    """
    from cobra.flux_analysis import flux_variability_analysis
    
    if reactions is None:
        reactions = [r.id for r in model.reactions]
    
    fva_result = flux_variability_analysis(model, reaction_list=reactions, 
                                           fraction_of_optimum=fraction_of_optimum)
    return fva_result

def identify_essential_genes(model, threshold: float = 0.01):
    """
    Identify essential genes in the model.
    
    :param model: The metabolic model
    :param threshold: Growth rate threshold to consider a gene essential
    :return: List of essential gene IDs
    """
    from cobra.flux_analysis import single_gene_deletion
    
    deletion_results = single_gene_deletion(model)
    essential_genes = deletion_results[deletion_results['growth'] < threshold].index.tolist()
    return essential_genes

def simulate_medium_changes(model, medium_changes: Dict[str, float]):
    """
    Simulate changes in the growth medium.
    
    :param model: The metabolic model
    :param medium_changes: Dictionary of exchange reaction IDs and their new bounds
    :return: Updated model
    """
    for rxn_id, bound in medium_changes.items():
        if rxn_id in model.reactions:
            model.reactions.get_by_id(rxn_id).bounds = (-bound, bound)
        else:
            print(f"Warning: Reaction {rxn_id} not found in the model.")
    return model

def calculate_yield(model, product_id: str, substrate_id: str):
    """
    Calculate the theoretical yield of a product from a substrate.
    
    :param model: The metabolic model
    :param product_id: ID of the product reaction
    :param substrate_id: ID of the substrate reaction
    :return: Yield value
    """
    model.objective = product_id
    model.optimize()
    product_flux = model.reactions.get_by_id(product_id).flux
    substrate_flux = abs(model.reactions.get_by_id(substrate_id).flux)
    
    if substrate_flux == 0:
        return 0
    return product_flux / substrate_flux

def add_metabolite_drain(model, metabolite_id: str, lb: float = 0, ub: float = 1000):
    """
    Add a drain reaction for a specific metabolite.
    
    :param model: The metabolic model
    :param metabolite_id: ID of the metabolite to add a drain for
    :param lb: Lower bound of the drain reaction
    :param ub: Upper bound of the drain reaction
    :return: Updated model
    """
    if metabolite_id not in model.metabolites:
        raise ValueError(f"Metabolite {metabolite_id} not found in the model.")
    
    drain_reaction = Reaction(f"DM_{metabolite_id}")
    drain_reaction.name = f"Drain of {metabolite_id}"
    drain_reaction.lower_bound = lb
    drain_reaction.upper_bound = ub
    
    metabolite = model.metabolites.get_by_id(metabolite_id)
    drain_reaction.add_metabolites({metabolite: -1})
    
    model.add_reactions([drain_reaction])
    return model

def compare_flux_distributions(flux_dist1: pd.Series, flux_dist2: pd.Series, threshold: float = 1e-6):
    """
    Compare two flux distributions and identify significant differences.
    
    :param flux_dist1: First flux distribution
    :param flux_dist2: Second flux distribution
    :param threshold: Threshold for considering a difference significant
    :return: DataFrame of significant flux differences
    """
    diff = flux_dist1 - flux_dist2
    significant_diff = diff[abs(diff) > threshold]
    result = pd.DataFrame({
        'flux1': flux_dist1[significant_diff.index],
        'flux2': flux_dist2[significant_diff.index],
        'difference': significant_diff
    })
    return result.sort_values('difference', key=abs, ascending=False)