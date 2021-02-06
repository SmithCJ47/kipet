"""Example 9: Data generation and estimability analysis

"""
# Standard library imports
import sys # Only needed for running the example from the command line

# Third party imports

# Kipet library imports
from kipet import KipetModel
from kipet.library.core_methods.EstimabilityAnalyzer import EstimabilityAnalyzer

if __name__ == "__main__":
    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False       
   
    """Simulation Model for generating data""" 
   
    kipet_model = KipetModel()
    
    sim_model = kipet_model.new_reaction('simulation')
    
    sim_model.add_component('A', value=0.5)
    sim_model.add_component('B', value=0.0)
    sim_model.add_component('C', value=0.0)
    sim_model.add_component('D', value=0.01)
    sim_model.add_component('E', value=0.0)
    sim_model.add_component('F', value=0.3)
    sim_model.add_component('G', value=0.5)
    sim_model.add_component('H', value=0.0)
    
    #Following this we add the kinetic parameters
    sim_model.add_parameter('k1', value=0.3)
    sim_model.add_parameter('k2', value=0.1)
    sim_model.add_parameter('k3', value=0.1)
    sim_model.add_parameter('k4', value=0.4)
    sim_model.add_parameter('k5', value=0.02)
    sim_model.add_parameter('k6', value=0.5)
    
    c = sim_model.get_model_vars()
    
    sim_model.add_ode('A', -c.k1*c.A - c.k4*c.A - c.k5*c.E*c.A )
    sim_model.add_ode('B',  c.k1*c.A - c.k2*c.B - c.k3*c.B )
    sim_model.add_ode('C',  c.k2*c.B - c.k4*c.C )
    sim_model.add_ode('D',  c.k4*c.A - c.k3*c.D )
    sim_model.add_ode('E',  c.k3*c.B - c.k5*c.E*c.A )
    sim_model.add_ode('F',  c.k5*c.E*c.A - c.k6*c.G**2*c.F )
    sim_model.add_ode('G', -c.k6*c.G**2*c.F )
    sim_model.add_ode('H',  c.k6*c.G**2*c.F )
    
    # sim_model.add_equations(rule_odes)
    sim_model.set_times(0, 20)
    sim_model.simulate()
    sim_model.plot('Z')

    # Add some noise and save the data
    data = kipet_model.add_noise_to_data(sim_model.results.Z, 0.02)
    filename = 'example_data/sim_data.csv'
    kipet_model.write_data_file(filename, data)
    
    """Make the model for estimability analysis"""
    
    # Clone the simulation model for the estimability analysis
    r1 = kipet_model.new_reaction('reaction-1', model=sim_model, items='model')
    
    # Add the generated data
    r1.add_data('C_data', category='concentration', file=filename)

    # Change the parameter initial values and add bounds
    new_inits = {'k1': 0.2,
                 'k2': 0.2,
                 'k3': 0.05,
                 'k4': 0.5,
                 'k5': 0.032,
                 'k6': 0.45,
                 }
    
    new_bounds = {k: (0, 1) for k in r1.parameters.names}
    
    r1.parameters.update('value', new_inits)
    r1.parameters.update('bounds', new_bounds)
    
    r1.create_pyomo_model()

    """EstimabilityAnalyzer - this has not been updated completely"""
    
    # Here we use the estimability analysis tools
    e_analyzer = EstimabilityAnalyzer(r1.model)
    # Problem needs to be discretized first
    e_analyzer.apply_discretization('dae.collocation',nfe=50,ncp=3,scheme='LAGRANGE-RADAU')
    # define the uncertainty surrounding each of the parameters
    # This is used for scaling the variables (i.e. 0.01 means that we are sure that the initial 
    # value ofthat parameter is within 1 % of the real value)
    param_uncertainties = {'k1':0.8,'k2':1.2,'k3':0.8,'k4':0.4, 'k5':1,'k6':0.3}
    # sigmas, as before, represent the variances in regard to component
    sigmas = {'A':1e-10,'B':1e-10,'C':1e-11, 'D':1e-11,'E':1e-11,'F':1e-11,'G':1e-11,'H':1e-11,'device':0.02}
    # measurement scaling
    meas_uncertainty = 0.02
    # The rank_params_yao function ranks parameters from most estimable to least estimable 
    # using the method of Yao (2003). Notice the required arguments. Returns a dictionary of rankings.
    listparams = e_analyzer.rank_params_yao(meas_scaling = meas_uncertainty, param_scaling = param_uncertainties, sigmas =sigmas)
    print(listparams)
    # Now we can run the analyzer using the list of ranked parameters
    params_to_select = e_analyzer.run_analyzer(method = 'Wu', parameter_rankings = listparams,meas_scaling = meas_uncertainty, variances =sigmas)
    # We can then use this information to fix certain parameters and run the parameter estimation
    print("The parameters that can be estimated are:", params_to_select)
    
    params_to_fix = list(set(r1.parameters.names).difference(params_to_select))
    
    
    """Run the PE again"""
    
    # Clone the simulation model without the model
    final_model = kipet_model.new_reaction('final', model=sim_model, items='model')

    # Add bounds to the parameter variables and change k5 to 0.032
    final_model.parameters.update('bounds', new_bounds)
    final_model.parameters.update('value', {'k5' : 0.032})
    
    # Update the component variances provided above
    final_model.components.update('variance', sigmas)
    
    # Add the experimental data
    final_model.add_data('C_data', category='concentration', file=filename)
    
    # Settings
    final_model.settings.parameter_estimator.solver = 'k_aug'
    
    # Fix the parameter that cannot be estimated here:
    final_model.fix_parameter(params_to_fix)
    
    # Run the parameter estimation
    final_model.run_opt()
    
    # Results and plot
    final_model.results.show_parameters
    if with_plots:
        final_model.plot()