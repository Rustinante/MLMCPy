import numpy as np
import timeit
from mpi4py import MPI

from spring_mass_model import SpringMassModel
from MLMCPy.input import RandomInput
from MLMCPy.mlmc import MLMCSimulator
from MLMCPy.mlmc import MLMCSimulator2

'''
This script demonstrates MLMCPy for simulating a spring-mass system with a 
random spring stiffness to estimate the expected value of the maximum 
displacement using multi-level Monte Carlo. Here, we use Model and RandomInput
objects with functional forms as inputs to MLMCPy. See the
/examples/spring_mass/from_data/ for an example of using precomputed data
in files as inputs.

For 8 processes, run from command line with something like:
mpiexec -n 8 python run_parallel_mlmc_from_model.py 
'''

# Set up MPI communicator for running in parallel
comm = MPI.COMM_WORLD.Clone()
rank = comm.Get_rank()   #this processors number/identifier (int) 
size = comm.Get_size()   #total number of processors

##########################################
# Step 1 - Establish Setup Parameters

## Global Input Parameters
num_samples = 5000  #for Monte Carlo only
low_timestep = 1.0
mid_timestep = 0.1
high_timestep = 0.01  #used by Monte Carlo

# Define random variable for spring stiffness:
# Need to provide a sampleable function to create RandomInput instance in MLMCPy
def beta_distribution(shift, scale, alpha, beta, size):
    return shift + scale*np.random.beta(alpha, beta, size)

np.random.seed(1)
stiffness_distribution = RandomInput(distribution_function=beta_distribution,
                                     shift=1.0, scale=2.5, alpha=3., beta=2.,
                                     random_seed=1)

##########################################
# Step 2: Serial Monte Carlo
# This will generate a reference solution and target 
# precision.

if rank == 0:
    print "################# TEST 1 ###################"
    ser_mc_total_cost = timeit.default_timer()
    model = SpringMassModel(mass=1.5, time_step=high_timestep)
    input_samples = stiffness_distribution.draw_samples(num_samples)
    output_samples_mc = np.zeros(num_samples)

    start_mc = timeit.default_timer()

    for i, sample in enumerate(input_samples):
        output_samples_mc[i] = model.evaluate([sample])

    ser_mc_computational_cost = timeit.default_timer() - start_mc

    ser_mean_mc = np.mean(output_samples_mc)
    ser_mc_total_cost = timeit.default_timer() - ser_mc_total_cost
    print "\nSUMMARY OF SERIAL MONTE CARLO:"
    print "Mean estimate from Serial Monte Carlo: ", ser_mean_mc
    ser_precision_mc = (np.var(output_samples_mc) / float(num_samples))
    print "Target precision from Serial Monte Carlo: ", ser_precision_mc
    print "Serial Monte Carlo Computational Cost: ", ser_mc_computational_cost
    print "Serial Monte Carlo Total Cost:         ", ser_mc_total_cost


else:
    # define dummy variables
    num_samples=None
    model=None
    input_samples=None
    output_samples=None
    start_mc=None
    output_samples_mc=None
    ser_mc_total_cost=None
    ser_mean_mc=None
    ser_precision_mc=None

comm.Barrier()


##########################################
# Step 3: Parallel Monte Carlo
# Run parallelized standard Monte Carlo to generate a reference solution and target 
# precision.  This is to ensure that the parallel output matches the serial output.

# This is done through a simple (though static) algorithm.
# Reference 1: https://stackoverflow.com/questions/36025188/along-what-axis-does-mpi4py-scatterv-function-split-a-numpy-array/36082684#36082684
# Reference 2: https://materials.jeremybejarano.com/MPIwithPython/collectiveCom.html
# TODO - try dynamic scheduling?

if rank == 0:
    print "\n################# TEST 2 ###################"
    par_mc_total_cost = timeit.default_timer()

stiffness_distribution.reset_sampling()
model = SpringMassModel(mass=1.5, time_step=high_timestep)

if rank == 0:
    input_samples = stiffness_distribution.draw_samples(num_samples)

    #Split input array by the number of available cores
    split_input_data = np.array_split(input_samples, size, axis=0)

    # create an array of the number of input samples each core gets
    split_send_counts = np.zeros(0, int)
    for i in range(0,len(split_input_data),1):
        split_send_counts = np.append(split_send_counts, len(split_input_data[i]))

    # make an array containing the number of elements away from the first
    # element in the array as which to begin the new, segemented array.
    split_displacements = np.insert(np.cumsum(split_send_counts),0,0)[0:-1]

else:
    #initialize the variables on the other ranks
    input_samples=None
    split_input_data=None
    split_send_counts=None
    split_displacements=None

# Broadcast the arrays to all of the ranks
split_send_counts = comm.bcast(split_send_counts, root=0)
split_displacements = comm.bcast(split_displacements, root=0)

#create arrays to store this ranks's subset of the data
this_rank_input_samples = np.zeros(split_send_counts[rank])
this_rank_output_samples = np.zeros(split_send_counts[rank])

# Scatter the data to all of the CPUs
comm.Scatterv([input_samples, tuple(split_send_counts), 
                tuple(split_displacements), MPI.DOUBLE], 
                this_rank_input_samples, root=0)

# Do the computations
local_start_mc = timeit.default_timer()

for i, sample in enumerate(this_rank_input_samples):
    this_rank_output_samples[i] = model.evaluate([sample])

local_total_cost = np.array([timeit.default_timer() - local_start_mc])

# Gather the data back to rank 0
if rank == 0:
    output_samples_mc = np.zeros(num_samples)
else:
    output_samples_mc = None

comm.Gatherv(this_rank_output_samples, [output_samples_mc, 
                tuple(split_send_counts), tuple(split_displacements), 
                MPI.DOUBLE], root=0)

# Gather and Report summary statistics 
par_mc_max_cost=np.zeros(1)
par_mc_min_cost=np.zeros(1)
par_mc_sum_cost=np.zeros(1)
comm.Reduce(local_total_cost, par_mc_max_cost, op=MPI.MAX, root=0)
comm.Reduce(local_total_cost, par_mc_min_cost, op=MPI.MIN, root=0)
comm.Reduce(local_total_cost, par_mc_sum_cost, op=MPI.SUM, root=0)

if rank == 0:
    print "\nSUMMARY OF PARALLEL MONTE CARLO:"
    par_mean_mc = np.mean(output_samples_mc)
    par_mc_total_cost = timeit.default_timer() - par_mc_total_cost
    print "Mean of the Monte Carlo output samples: ", par_mean_mc
    par_precision_mc = (np.var(output_samples_mc) / float(num_samples))
    print "Target precision: ", par_precision_mc

    print "Max single-cpu computation time: ", par_mc_max_cost[0]
    print "Min single-cpu computation time: ", par_mc_min_cost[0]
    print "Avg single-cpu computation time: ", par_mc_sum_cost[0]/size
    print "Total Parallel MC time: ", par_mc_total_cost
else:
    par_mean_mc=None
    par_precision_mc=None

mean_mc = comm.bcast(par_mean_mc, root=0)
precision_mc = comm.bcast(par_precision_mc, root=0)

comm.Barrier()

#####################################################
# Step 4: SERIAL - Original MLMCPy Monte Carlo Simulation
# Use to generate a baseline reference for time and accuracy

if rank == 0:
    print "\n################# TEST 3 ###################"
    print "SUMMARY OF SERIAL MLMCPY:"
    print "PLEASE USE SEPARATE RUN SCRIPT."
    print "CANNOT RUN SERIAL MLMCPY IN SAME SCRIPT AS" 
    print "PARALLEL MLMCPY. Sorry!\n"

comm.Barrier()
#####################################################
# Step 4: PARALLEL - Original MLMCPy Monte Carlo Simulation
# Use to generate a baseline reference for time and accuracy

stiffness_distribution.reset_sampling()

if rank == 0:
    print "\n################# TEST 4 ###################"
    print "Starting Parallel MLMCPy comparisons..."
    orig_mlmc_total_cost = timeit.default_timer()

# Initialize spring-mass models for MLMC. Here using three levels 
# with MLMC defined by different time steps
model_level1 = SpringMassModel(mass=1.5, time_step=low_timestep)
model_level2 = SpringMassModel(mass=1.5, time_step=mid_timestep)
model_level3 = SpringMassModel(mass=1.5, time_step=high_timestep)

models = [model_level1, model_level2, model_level3]

# Initialize MLMC & predict max displacement to specified precision
mlmc_simulator = MLMCSimulator(stiffness_distribution, models)

local_mlmc_cost = timeit.default_timer()

[estimates, sample_sizes, variances] = \
    mlmc_simulator.simulate(epsilon=np.sqrt(precision_mc),
                            initial_sample_sizes=100,
                            verbose=True)

local_mlmc_cost = np.array([timeit.default_timer() - local_mlmc_cost])

orig_mlmc_max_cost=np.zeros(1)
orig_mlmc_min_cost=np.zeros(1)
orig_mlmc_sum_cost=np.zeros(1)
comm.Reduce(local_mlmc_cost, orig_mlmc_max_cost, op=MPI.MAX, root=0)
comm.Reduce(local_mlmc_cost, orig_mlmc_min_cost, op=MPI.MIN, root=0)
comm.Reduce(local_mlmc_cost, orig_mlmc_sum_cost, op=MPI.SUM, root=0)

#Summarize results:
if rank == 0:
    orig_mlmc_total_cost = timeit.default_timer() - orig_mlmc_total_cost
    print "\nSUMMARY OF ORIGINAL PARALLEL MLMCPY:"
    print 'Original MLMC sample sizes used: ', sample_sizes
    print 'Original MLMC estimate: %s' % estimates[0]
    print 'Original MLMC precision: %s' % variances[0]

    print "Max single-cpu computation time: ", orig_mlmc_max_cost[0]
    print "Min single-cpu computation time: ", orig_mlmc_min_cost[0]
    print "Avg single-cpu computation time: ", orig_mlmc_sum_cost[0]/size
    print 'Original MLMC total cost: %s' % orig_mlmc_total_cost

comm.Barrier()

#####################################################
# Step 5: PARALLEL - Original MLMCPy Monte Carlo Simulation
# Use to generate a baseline reference of parallel timing

stiffness_distribution.reset_sampling()

if rank == 0:
    print "\n################# TEST 5 ###################"
    NEW_mlmc_total_cost = timeit.default_timer()

# Initialize spring-mass models for MLMC. Here using three levels 
# with MLMC defined by different time steps
NEW_model_level1 = SpringMassModel(mass=1.5, time_step=low_timestep)
NEW_model_level2 = SpringMassModel(mass=1.5, time_step=mid_timestep)
NEW_model_level3 = SpringMassModel(mass=1.5, time_step=high_timestep)

NEW_models = [model_level1, model_level2, model_level3]

# Initialize MLMC & predict max displacement to specified precision
NEW_mlmc_simulator = MLMCSimulator2(stiffness_distribution, NEW_models)

NEW_local_mlmc_cost = timeit.default_timer()

[NEW_estimates, NEW_sample_sizes, NEW_variances] = \
    NEW_mlmc_simulator.simulate(epsilon=np.sqrt(precision_mc),
                            initial_sample_sizes=100,
                            verbose=True)

NEW_local_mlmc_cost = np.array([timeit.default_timer() - NEW_local_mlmc_cost])

NEW_mlmc_max_cost=np.zeros(1)
NEW_mlmc_min_cost=np.zeros(1)
NEW_mlmc_sum_cost=np.zeros(1)
comm.Reduce(NEW_local_mlmc_cost, NEW_mlmc_max_cost, op=MPI.MAX, root=0)
comm.Reduce(NEW_local_mlmc_cost, NEW_mlmc_min_cost, op=MPI.MIN, root=0)
comm.Reduce(NEW_local_mlmc_cost, NEW_mlmc_sum_cost, op=MPI.SUM, root=0)

#Summarize results:
if rank == 0:
    NEW_mlmc_total_cost = timeit.default_timer() - NEW_mlmc_total_cost
    print "\nSUMMARY OF NEW PARALLEL MLMCPY:"
    print 'NEW MLMC sample sizes used: ', sample_sizes
    print 'NEW MLMC estimate: %s' % estimates[0]
    print 'NEW MLMC precision: %s' % variances[0]

    print "Max single-cpu computation time: ", NEW_mlmc_max_cost[0]
    print "Min single-cpu computation time: ", NEW_mlmc_min_cost[0]
    print "Avg single-cpu computation time: ", NEW_mlmc_sum_cost[0]/size
    print 'NEW MLMC total cost: %s' % NEW_mlmc_total_cost

comm.Barrier()
#####################################################
# Step 6: Final speedup comparisons
# Display the overall speedup results from all runs
if rank == 0:
    print "\n################# SPEEDUP SUMMARY ###################"
    print "Serial MC vs. Parallel MC Total speedup: ", ser_mc_total_cost / par_mc_total_cost
    print "Serial MC vs. Parallel MC Computational speedup: ", ser_mc_computational_cost / par_mc_max_cost[0]
    print
    print "Parallel MC vs. Original MLMC Total Speedup: ", par_mc_total_cost / orig_mlmc_total_cost
    print "Parallel MC vs. Original MLMC Computational Speedup: ", par_mc_max_cost[0] / orig_mlmc_max_cost[0]
    print
    print "Parallel MC vs. NEW MLMC Total Speedup: ", par_mc_total_cost / NEW_mlmc_total_cost
    print "Parallel MC vs. NEW MLMC Computational Speedup: ", par_mc_max_cost[0] / NEW_mlmc_max_cost[0]
    print
    print "Original MLMC vs. NEW MLMC Total Speedup: ", orig_mlmc_total_cost / NEW_mlmc_total_cost
    print "Original MLMC vs. NEW MLMC Computational Speedup: ", orig_mlmc_max_cost[0] / NEW_mlmc_max_cost[0]





