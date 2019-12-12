import numpy as np
import timeit
from MLMCPy.scheduler import get_job_allocation_heuristically
from datetime import timedelta
import imp

try:
    from mpi4py import MPI
except:
    pass

from MLMCPy.input import Input
from MLMCPy.model import Model


class MLMCSimulator:
    """
    Computes an estimate based on the Multi-Level Monte Carlo algorithm.
    """

    def __init__(self, data, models, use_original_mlmc=True):
        """
        Requires a data object that provides input samples and a list of models
        of increasing fidelity.

        :param data: Provides a data sampling function.
        :type data: Input
        :param models: Each model Produces outputs from sample data input.
        :type models: list(Model)
        """
        # Detect whether we have access to multiple CPUs.
        self._detect_parallelization()

        self.data = data
        self.models = models
        self.num_levels = len(self.models)
        self._check_init_parameters(data, models)

        # Use the original MLMC code (or an optimized parallelization version)
        self.use_original_mlmc = use_original_mlmc

        # Sample size to be taken at each level.
        self.sample_sizes = np.zeros(self.num_levels, dtype='int')

        # Used to compute sample sizes based on a fixed cost.
        self._target_cost = None

        # Sample sizes used in setup.
        self.initial_sample_sizes = None

        # Desired level of precision.
        self._epsilons = np.zeros(1)

        # Number of elements in model output.
        test_sample = self.reset_draw_reset(self.num_cpus)[0]
        self.input_size = test_sample.size
        self.output_size = self.models[0].evaluate(test_sample).size

        self.cached_inputs = None
        self.cached_outputs = None

        # Whether to allow use of model output caching.
        self.caching_enabled = False

        # Enabled diagnostic text output.
        self.verbose = False

    def simulate(self, epsilon, initial_sample_sizes=100, target_cost=None, sample_sizes=None, verbose=False):
        """
        Perform MLMC simulation.
        Computes number of samples per level before running simulations
        to determine estimates.
        Can be specified based on target precision to achieve (epsilon), 
        total target cost (in seconds), or on number of sample to run on each 
        level directly.

        :param epsilon: Desired accuracy to be achieved for each quantity of interest.
        :type epsilon: float, list of floats, or ndarray.

        :param initial_sample_sizes: Sample sizes used when computing cost
            and variance for each model in simulation.
        :type initial_sample_sizes: ndarray, int, list

        :param target_cost: Target cost to run simulation (optional).
            If specified, overrides any epsilon value provided.
        :type target_cost: float or int

        :param np.ndarray sample_sizes: Number of samples to compute at each level
        :param bool verbose: Whether to print useful diagnostic information.
        :return: tuple of np.ndarray
            (estimates, sample count per level, variances)
        """
        self.verbose = verbose and self.cpu_rank == 0
        self.set_target_cost(target_cost)
        self.reset_data_sampling()

        if sample_sizes is None:
            self.initial_sample_sizes = self._convert_sample_sizes_to_np_array(initial_sample_sizes)

        cost_var_estimates = self.setup_simulation(epsilon, sample_sizes)

        return self._run_simulation(cost_var_estimates)

    def setup_simulation(self, epsilon, sample_sizes):
        """
        Performs any necessary manipulation of epsilon and initial_sample_sizes.
        Computes variance and cost at each level in order to estimate optimal
        number of samples at each level.

        :param epsilon: Epsilon values for each quantity of interest.
        """
        if sample_sizes is None:
            self.set_epsilon(epsilon)

            costs, variances = self._compute_costs_and_variances()
            self._compute_optimal_sample_sizes(costs, variances)
            self.caching_enabled = False
            return costs, variances

        else:
            self._target_cost = None
            self.caching_enabled = False
            sample_sizes = self._convert_sample_sizes_to_np_array(sample_sizes, initial_samples=False)
            self._process_sample_sizes(sample_sizes, None)

    def _compute_costs_and_variances(self):
        """
        Compute costs and variances across levels.

        :return: tuple of ndarrays:
            1d ndarray of costs
            2d ndarray of variances
        """
        if self.verbose:
            print("Determining costs: ")

        self._initialize_cache()

        # Evaluate samples in model. Gather compute times for each level.
        # Variance is computed from difference between outputs of adjacent
        # layers evaluated from the same samples.
        compute_times = np.zeros(self.num_levels)

        for level in range(self.num_levels):
            input_samples = self._draw_setup_samples(level)

            start_time = timeit.default_timer()
            self._compute_setup_outputs(input_samples, level)
            compute_times[level] = timeit.default_timer() - start_time

        # Get outputs across all CPUs before computing variances.
        if not self.use_original_mlmc:
            if self.num_cpus == 1:
                all_outputs = self.cached_outputs
            else:
                all_outputs_tmp = np.zeros(
                    (self.num_levels * self.initial_sample_sizes[0], self.output_size)
                )

                # create count vector
                num_residual_samples = self.initial_sample_sizes[0] % self.num_cpus
                counts = np.ones(self.num_cpus, dtype='int') * (self.initial_sample_sizes[0] // self.num_cpus)
                for rank in range(num_residual_samples):
                    counts[rank] += 1

                # create displacements vector
                displ = np.insert(np.cumsum(counts), 0, 0)[:-1]

                # augment arrays to account for there being multiple levels
                counts_buf = counts * self.num_levels * self.output_size
                displ_buf = displ * self.num_levels * self.output_size

                # self._cached_outputs have shape (num_levels, num_cpu_samples, output_size)
                # Allgatherv the cache samples to each cpu
                self._comm.Allgatherv(
                    [self.cached_outputs, MPI.DOUBLE],
                    [all_outputs_tmp, tuple(counts_buf), tuple(displ_buf), MPI.DOUBLE]
                )

                # unpack the gathered array into the correct format
                all_outputs = np.zeros((self.num_levels,
                                        self.initial_sample_sizes[0],
                                        self.output_size))

                for rank, num_cpu_samples in enumerate(counts):
                    cpu_start_index = displ[rank] * self.num_levels
                    for level in range(self.num_levels):
                        level_start_index = cpu_start_index + num_cpu_samples * level
                        level_end_index = level_start_index + num_cpu_samples

                        all_outputs[level, displ[rank]:displ[rank] + counts[rank], :] = \
                            all_outputs_tmp[level_start_index:level_end_index, :]

        else:
            all_outputs = self._gather_arrays(self.cached_outputs, axis=1)

        variances = np.var(all_outputs, axis=1)
        costs = self._compute_costs(compute_times)

        if self.verbose:
            print('Initial sample variances: \n%s' % variances)

        return costs, variances

    def _initialize_cache(self):
        """
        Sets up the cache for retaining model outputs evaluated in the setup
        phase for reuse in the simulation phase.
        """
        # Determine number of samples to be taken on this processor.
        # (this function gets the num sample sizes per level, and then
        # expands that single number so that a copy of it exists for each level
        # i.e. it turns 12 into [12 12 12]
        self._cpu_initial_sample_sizes = np.vectorize(self._determine_num_cpu_samples)(self.initial_sample_sizes)

        cpu_sample_size = np.max(self._cpu_initial_sample_sizes)
        # Cache model outputs computed here so that they can be reused
        # in the simulation.
        self.cached_inputs = np.zeros((self.num_levels, cpu_sample_size, self.input_size))
        self.cached_outputs = np.zeros((self.num_levels, cpu_sample_size, self.output_size))

    def _draw_setup_samples(self, level):
        """
        Draw samples based on initial sample size at specified level.
        Store samples in _cached_inputs.
        :param level: int level
        """
        input_samples = self._draw_samples(self.initial_sample_sizes[level])

        # To cache these samples, we have to account for the possibility
        # of the data source running out of samples so that we can
        # broadcast into the cache successfully.
        self.cached_inputs[level, :input_samples.shape[0], :] = input_samples

        return input_samples

    def _compute_setup_outputs(self, input_samples, level):
        """
        Evaluate model outputs for a given level. If level > 0, subtract outputs
        at level below specified level. Store results in _cached_outputs.
        :param input_samples: samples to evaluate in model.
        :param level: int level of model
        """
        for i, sample in enumerate(input_samples):
            self.cached_outputs[level, i] = self.models[level].evaluate(sample)
            if level > 0:
                self.cached_outputs[level, i] -= self.models[level - 1].evaluate(sample)

    def _compute_costs(self, compute_times):
        """
        Set costs for each level, either from precomputed values from each
        model or based on computation times provided by compute_times.

        :param compute_times: ndarray of computation times for computing
        model at each layer and preceding layer.
        """
        # If the models have costs predetermined, use them to compute costs
        # between each level.
        if self._models_have_costs():
            costs = self._get_costs_from_models()
        else:
            # Compute costs based on compute time differences between levels.
            costs = compute_times / self._cpu_initial_sample_sizes  # \
            #  * self.num_cpus

        costs = self._mean_over_all_cpus(costs)

        if self.verbose:
            print(np.array2string(costs))

        return costs

    def _models_have_costs(self):
        """
        :return: bool indicating whether the models all have a cost attribute.
        """
        model_cost_defined = True
        for model in self.models:

            model_cost_defined = model_cost_defined and hasattr(model, 'cost')

            if not model_cost_defined:
                return False

            model_cost_defined = model_cost_defined and model.cost is not None

        return model_cost_defined

    def _get_costs_from_models(self):
        """
        Collect cost value from each model.
        :return: ndarray of costs.
        """
        costs = np.ones(self.num_levels)
        for i, model in enumerate(self.models):
            costs[i] = model.cost

        # Costs at level > 0 should be summed with previous level.
        costs[1:] = costs[1:] + costs[:-1]

        return costs

    def _compute_optimal_sample_sizes(self, costs, variances):
        """
        Compute the sample size for each level to be used in simulation.

        :param variances: 2d ndarray of variances
        :param costs: 1d ndarray of costs
        """
        if self.verbose:
            print("Computing optimal sample sizes: ")

        # Need 2d version of costs in order to vectorize the operations.
        costs = costs[:, np.newaxis]

        mu = self._compute_mu(costs, variances)

        # Compute sample sizes.
        sqrt_v_over_c = np.sqrt(variances / costs)
        sample_sizes = np.amax(np.trunc(mu * sqrt_v_over_c), axis=1)

        self._process_sample_sizes(sample_sizes, costs)

        if self.verbose:
            print(np.array2string(self.sample_sizes))

            estimated_runtime = np.dot(self.sample_sizes, np.squeeze(costs))

            self._show_time_estimate(estimated_runtime)

    def _compute_mu(self, costs, variances):
        """
        Computes the mu value used to compute sample sizes.

        :param costs: 2d ndarray of costs
        :param variances: ndarray of variances
        :return: ndarray of mu value for each QoI.
        """

        if self._target_cost is None:
            sum_sqrt_vc = np.sum(np.sqrt(variances * costs), axis=0)
            mu = np.power(self._epsilons, -2) * sum_sqrt_vc
        else:
            max_variances = np.max(variances, axis=1).reshape(costs.shape)
            sum_sqrt_vc = np.sum(np.sqrt(max_variances * costs), axis=0)
            mu = self._target_cost * float(self.num_cpus) / sum_sqrt_vc

        return mu

    def _process_sample_sizes(self, sample_sizes, costs):
        """
        Make any necessary adjustments to computed sample sizes, including
        adjustments for staying under target cost and distributing among
        processors.
        """
        self.sample_sizes = sample_sizes

        # Manually tweak sample sizes to get predicted cost closer to target.
        if self._target_cost is not None:
            self._fit_samples_sizes_to_target_cost(np.squeeze(costs))

        # Set sample sizes to ints.
        self.sample_sizes = self.sample_sizes.astype(int)

        # If target cost is less than cost of least expensive model, run it
        # once so we are at least doing something in the simulation.
        if np.sum(self.sample_sizes) == 0.:
            self.sample_sizes[0] = 1

        # Divide sampling evenly across CPUs.
        split_samples = np.vectorize(self._determine_num_cpu_samples)
        self._cpu_sample_sizes = split_samples(self.sample_sizes)

    def _fit_samples_sizes_to_target_cost(self, costs):
        """
        Adjust sample sizes to be as close to the target cost as possible.
        """
        # Find difference between projected total cost and target.
        total_cost = np.dot(costs, self.sample_sizes)
        difference = self._target_cost - total_cost
        # If the difference is greater than the lowest cost model, adjust
        # the sample sizes.
        if abs(difference) > costs[0]:

            # Start with highest cost model and add samples in order to fill
            # the cost gap as much as possible.
            for i in range(len(costs) - 1, -1, -1):
                if costs[i] < abs(difference):

                    # Compute number of samples we can fill the gap with at
                    # current level.
                    delta = np.trunc(difference / costs[i])
                    self.sample_sizes[i] += delta

                    if self.sample_sizes[i] < 0:
                        self.sample_sizes[i] = 0

                    # Update difference from target cost.
                    total_cost = np.sum(costs * self.sample_sizes)
                    difference = self._target_cost - total_cost

    def _run_simulation(self, cost_var_estimates=None):
        """
        Compute estimate by extracting number of samples from each level
        determined in the setup phase.

        :return: tuple containing three ndarrays:
            estimates: Estimates for each quantity of interest.
            sample_sizes: The sample sizes used at each level.
            variances: Variance of model outputs at each level.
        """
        # Sampling needs to be restarted from beginning due to sampling
        # having been performed in setup phase.
        self.data.reset_sampling()

        start_time = timeit.default_timer()
        estimates, variances = self._run_simulation_loop(cost_var_estimates)
        run_time = timeit.default_timer() - start_time

        if self.verbose:
            self._show_summary_data(estimates, variances, run_time)

        return estimates, self.sample_sizes, variances

    def _run_simulation_loop(self, cost_var_estimates):
        """
        Main simulation loop where sample sizes determined in setup phase are
        drawn from the input data and run through the models. Values for
        computing the estimates and variances are accumulated at each level.

        :return: tuple containing two ndarrays:
            estimates: Estimates for each quantity of interest.
            variances: Variance of model outputs at each level.
        """
        estimates, variances = self.get_zero_initialized_estimates_and_variances()

        if not self.use_original_mlmc:
            np.random.seed(self.cpu_rank)
            if cost_var_estimates is None:
                raise ValueError('must provide cost_var_estimates for the new mlmc')
            costs, _ = cost_var_estimates
            samples, cpu_to_num_samples = self._draw_samples_with_predetermined_sizes(costs=costs)
            level_sizes = cpu_to_num_samples[self.cpu_rank]

            # Generate all outputs for this cpu
            offset = 0
            cpu_outputs = np.zeros((np.sum(level_sizes), self.output_size))
            for level, s in enumerate(level_sizes):
                for i, x in enumerate(samples[offset:offset + s]):
                    if level == 0:
                        cpu_outputs[offset + i] = self.models[level].evaluate(x)
                    else:  # level > 0
                        cpu_outputs[offset + i] = self.models[level].evaluate(x) - self.models[level - 1].evaluate(x)
                offset += s

            counts = np.sum(cpu_to_num_samples, axis=1)
            counts_buf = counts * self.output_size

            # create displacements tuple (already accounts for output dimensions)
            displ = np.insert(np.cumsum(counts), 0, 0)[:-1]
            displ_buf = displ * self.output_size

            all_outputs_flattened = np.zeros((np.sum(cpu_to_num_samples), self.output_size))

            start_gather = timeit.default_timer()
            self._comm.Gatherv([cpu_outputs, MPI.DOUBLE],
                               [all_outputs_flattened,
                                tuple(counts_buf),
                                tuple(displ_buf),
                                MPI.DOUBLE],
                               root=0)
            end_gather = timeit.default_timer() - start_gather

            num_samples_per_level = np.sum(cpu_to_num_samples, axis=0)
            start_unpack = timeit.default_timer()
            # unpack the gathered array into the correct format
            if self.cpu_rank == 0:
                for level in range(self.num_levels):
                    level_outputs = np.zeros((num_samples_per_level[level], self.output_size))

                    for rank in range(self.num_cpus):
                        all_out_start = displ[rank] + np.sum(cpu_to_num_samples[rank, :level])
                        all_out_end = all_out_start + cpu_to_num_samples[rank, level]

                        lvl_out_start = np.sum(cpu_to_num_samples[:rank, level])
                        lvl_out_end = lvl_out_start + cpu_to_num_samples[rank][level]

                        level_outputs[lvl_out_start:lvl_out_end] = all_outputs_flattened[all_out_start:all_out_end]

                    estimates += np.sum(level_outputs, axis=0) / num_samples_per_level[level]
                    variances += np.var(level_outputs, axis=0) / num_samples_per_level[level]

            end_unpack = timeit.default_timer() - start_unpack
            if self.cpu_rank == 0:
                print("time to gatherv: ", end_gather)
                print("time to unpack: ", end_unpack)
                print("total time to gather and process: ", end_gather + end_unpack)

        else:
            for level in range(self.num_levels):

                if self.sample_sizes[level] == 0:
                    continue

                samples = self._get_sim_loop_samples(level)
                output_differences = self._get_sim_loop_outputs(samples, level)
                self._update_sim_loop_values(
                    output_differences, level, acc_estimates=estimates, acc_variances=variances
                )

        return estimates, variances

    def _get_sim_loop_samples(self, level):
        """
        Acquires input samples for designated level.

        :param level: int of level for which samples are to be acquired.
        :return: ndarray of input samples.
        """
        samples = self._draw_samples(int(self.sample_sizes[level]))
        num_samples = samples.shape[0]

        # Update sample sizes in case we've run short on samples.
        self._cpu_sample_sizes[level] = num_samples

        return samples

    def _draw_samples_with_predetermined_sizes(self, costs):
        cpu_to_predetermined_sizes, _ = get_job_allocation_heuristically(
            job_counts=self.sample_sizes, costs=costs, num_workers=self.num_cpus
        )
        if self.cpu_rank == 0:
            print('cpu_to_predetermined_sizes: ')
            print(cpu_to_predetermined_sizes)
        num_samples_for_current_cpu = np.sum(cpu_to_predetermined_sizes[self.cpu_rank])
        samples = self.data.draw_samples(int(num_samples_for_current_cpu))
        return samples, cpu_to_predetermined_sizes

    def _get_sim_loop_outputs(self, samples, level):
        """
        Get the output differences for given level and samples.

        :param samples: ndarray of input samples.
        :param level: int level of model to run.

        :return: ndarray of output differences between samples from
            designated level and level below (if applicable).
        """
        num_samples = samples.shape[0]

        if num_samples == 0:
            return np.zeros((1, self.output_size))

        output_differences = np.zeros((num_samples, self.output_size))

        for i, sample in enumerate(samples):
            output_differences[i] = self._evaluate_sample(sample, level)

        return output_differences

    def _update_sim_loop_values(self, outputs, level, acc_estimates, acc_variances):
        """
        Update running totals for estimates and variances based on the output
        differences at a particular level.

        :param outputs: ndarray of output differences.
        :param level: int of level at which differences were computed.
        """
        if self.cpu_rank == 0:
            start_time = timeit.default_timer()

        cpu_samples = self._cpu_sample_sizes[level]

        all_output_differences = self._gather_arrays(outputs, axis=0)

        self.sample_sizes[level] = self._sum_over_all_cpus(cpu_samples)
        num_samples = float(self.sample_sizes[level])

        acc_estimates += np.sum(all_output_differences, axis=0) / num_samples
        acc_variances += np.var(all_output_differences, axis=0) / num_samples
        if self.cpu_rank == 0:
            end_time = timeit.default_timer() - start_time
            print("time to gather and process values: ", end_time)

    def _evaluate_sample(self, sample, level):
        """
        Evaluate output of an input sample, either by running the model or
        retrieving the output from the cache. For levels > 0, returns
        difference between current level and lower level outputs.

        :param sample: sample value
        :param level: model level
        :return: result of evaluation
        """
        sample_indices = np.empty(0)
        if self.caching_enabled:
            bool_array = (sample == self.cached_inputs[level])
            sample_indices = np.where((bool_array).all(axis=1))[0]
        if len(sample_indices) == 1:
            output = self.cached_outputs[level, sample_indices[0]]
        else:
            output = self.models[level].evaluate(sample)

            # If we are at a level greater than 0, compute outputs for lower
            # level and subtract them from this level's outputs.
            if level > 0:
                output -= self.models[level - 1].evaluate(sample)

        return output

    def _show_summary_data(self, estimates, variances, run_time):
        """
        Shows summary of simulation.

        :param estimates: ndarray of estimates for each QoI.
        :param variances: ndarray of variances for each QoI.
        """
        # Compare variance for each quantity of interest to epsilon values.
        print('Total run time: %s' % str(run_time))

        epsilons_squared = np.square(self._epsilons)
        for i, variance in enumerate(variances):
            passed = variance < epsilons_squared[i]
            estimate = estimates[i]

            print('QOI #%s: estimate: %s, variance: %s, ' \
                  'epsilon^2: %s, met target precision: %s' % \
                  (i, float(estimate), float(variance),
                   float(epsilons_squared[i]), passed))

    def reset_draw_reset(self, num_samples):
        self.data.reset_sampling()
        samples = self.data.draw_samples(num_samples)
        self.data.reset_sampling()
        return samples

    def reset_data_sampling(self):
        self.data.reset_sampling()

    def get_zero_initialized_estimates_and_variances(self):
        estimates = np.zeros(self.output_size)
        variances = np.zeros_like(estimates)
        return estimates, variances

    def set_epsilon(self, epsilon):
        """
        Produce an ndarray of epsilon values from scalar or vector of epsilons.
        If a vector, length should match the number of quantities of interest.

        :param epsilon: float, list of floats, or ndarray.
        """
        if isinstance(epsilon, list):
            epsilon = np.array(epsilon)

        if isinstance(epsilon, float):
            epsilon = np.ones(self.output_size) * epsilon

        if not isinstance(epsilon, np.ndarray):
            raise TypeError("Epsilon must be a float, list of floats, or an ndarray.")

        if np.any(epsilon <= 0.):
            raise ValueError("Epsilon values must be greater than 0.")

        if len(epsilon) != self.output_size:
            raise ValueError("Number of epsilons must match number of levels.")

        self._epsilons = epsilon

    def _convert_sample_sizes_to_np_array(self, sample_sizes, initial_samples=True):
        """
        Produce an array of sample sizes, ensuring that its length
        matches the number of models.
        :param sample_sizes: scalar or vector of sample sizes

        returns verified/adjusted sample sizes array
        """
        if isinstance(sample_sizes, np.ndarray):
            verified_sample_sizes = sample_sizes

        elif isinstance(sample_sizes, list):
            verified_sample_sizes = np.array(sample_sizes)

        else:
            if not isinstance(sample_sizes, (int, float, np.int32, np.int64, np.float32, np.float64)):
                raise TypeError("Initial sample sizes must be numeric.")

            verified_sample_sizes = np.ones(self.num_levels).astype(int) * int(sample_sizes)

        if verified_sample_sizes.size != self.num_levels:
            raise ValueError("Number of initial sample sizes must match the number of models.")

        if not np.all(verified_sample_sizes > 1) and initial_samples:
            raise ValueError("Each initial sample size must be at least 2.")

        return verified_sample_sizes

    def _check_init_parameters(self, data, models):
        if not isinstance(data, Input):
            raise TypeError("data must a subclass of Input.")

        for model in models:
            if not isinstance(model, Model):
                TypeError("models must be a list of models.")

        test_sample = self.reset_draw_reset(num_samples=1)[0]
        output_sizes = [m.evaluate(test_sample).size for m in models]
        if len(set(output_sizes)) > 1:
            raise ValueError('All models must have the same output dimension.')

    def set_target_cost(self, target_cost):
        """
        Inspect parameters to simulate method.
        :param target_cost: float or int specifying desired simulation cost.
        """
        if target_cost is not None:
            if not isinstance(target_cost, (float, int, np.int32, np.int64, np.float32, np.float64)):
                raise TypeError('maximum cost must be an int or float.')

            if target_cost <= 0:
                raise ValueError("maximum cost must be greater than zero.")

            self._target_cost = target_cost

    def _draw_samples(self, num_samples):
        """
        Draw samples from data source.
        :param num_samples: Total number of samples to draw over all CPUs.
        :return: ndarray of samples sliced according to number of CPUs.
        """
        samples = self.data.draw_samples(num_samples)
        if self.num_cpus == 1:
            return samples

        sample_size = samples.shape[0]

        # Determine subsample sizes for all CPUs.
        subsample_size = sample_size // self.num_cpus
        remainder = sample_size - subsample_size * self.num_cpus
        subsample_sizes = np.ones(self.num_cpus + 1).astype(int) * subsample_size

        # Adjust for sampling that does not divide evenly among CPUs.
        subsample_sizes[:remainder + 1] += 1
        subsample_sizes[0] = 0

        # Determine starting index of subsample.
        subsample_index = int(np.sum(subsample_sizes[:self.cpu_rank + 1]))

        samples = samples[subsample_index: subsample_index + subsample_sizes[self.cpu_rank + 1], :]

        return samples

    def _detect_parallelization(self):
        """
        Detects whether multiple processors are available and sets
        self.number_CPUs and self.cpu_rank accordingly.
        """
        try:
            imp.find_module('mpi4py')

            from mpi4py import MPI
            comm = MPI.COMM_WORLD

            self.num_cpus = comm.size
            self.cpu_rank = comm.rank
            self._comm = comm

        except ImportError:
            print('=> mpi4py not found')
            self.num_cpus = 1
            self.cpu_rank = 0

    def _mean_over_all_cpus(self, this_cpu_values, axis=0):
        """
        Finds the mean of ndarray of values across CPUs and returns result.
        :param this_cpu_values: ndarray of any shape.
        :return: ndarray of same shape as values with mean from all cpus.
        """
        if self.num_cpus == 1:
            return this_cpu_values

        if not self.use_original_mlmc:
            all_values = np.zeros(self.num_levels)
            self._comm.Allreduce(
                [this_cpu_values, MPI.DOUBLE],
                [all_values, MPI.DOUBLE],
                op=MPI.SUM
            )
            all_values /= self.num_cpus
            return all_values

        else:
            all_values = self._comm.allgather(this_cpu_values)
            return np.mean(all_values, axis)

    def _sum_over_all_cpus(self, this_cpu_values, axis=0):
        """
        Collect arrays from all CPUs and perform summation over specified
        axis.
        :param this_cpu_values: ndarray provided for current CPU.
        :param axis: int axis to perform summation over.
        :return: ndarray of summation result
        """
        if self.num_cpus == 1:
            return this_cpu_values

        all_values = self._comm.allgather(this_cpu_values)

        return np.sum(all_values, axis)

    def _gather_arrays(self, this_cpu_array, axis=0):
        """
        Collects an array from all processes and combines them so that single
        processor ordering is preserved.
        :param this_cpu_array: Arrays to be combined.
        :param axis: Axis to concatenate the arrays on.
        :return: ndarray
        """
        if self.num_cpus == 1:
            return this_cpu_array

        gathered_arrays = self._comm.allgather(this_cpu_array)

        return np.concatenate(gathered_arrays, axis=axis)

    def _determine_num_cpu_samples(self, total_num_samples):
        """
        Determines number of samples to be run on current cpu based on the total number of samples to run.
        :param total_num_samples: Total samples to be taken.
        :return: Samples to be taken by this cpu.
        """
        num_cpu_samples = total_num_samples // self.num_cpus
        num_residual_samples = total_num_samples % self.num_cpus

        if self.cpu_rank < num_residual_samples:
            num_cpu_samples += 1

        return num_cpu_samples

    @staticmethod
    def _show_time_estimate(seconds):
        """
        Used to show theoretical simulation time when verbose is enabled.
        :param seconds: int seconds to convert to readable time.
        """
        if isinstance(seconds, np.ndarray):
            seconds = seconds[0]

        time_delta = timedelta(seconds=seconds)

        print('Estimated simulation time: %s' % str(time_delta))
