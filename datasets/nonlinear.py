import numpy as np
import random
import igraph as ig
import os
from tqdm import tqdm


class Nonlinear:
    def __init__(self, options):
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.n = options['training_size'] + options['testing_size']
        self.t = options['T']
        self.m = options['m']
        self.num_vars = options['num_vars']
        self.data_dir = options['data_dir']
        self.mul = options['mul']
        self.adlength = options['adlength']
        self.adtype = options['adtype']
        self.noise_scale = options['noise_scale']
        self.dependent_features = options['dependent_features']
        self.generate_dag_graph()

    def generate_er_graph(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        # Generate a directed Erdös-Renyi graph and store its transposed adjacency matrix.
        ig.set_random_number_generator(random.Random(self.seed))
        G_und = ig.Graph.Erdos_Renyi(n=self.num_vars, m=self.m, directed=True, loops=False)
        self.data_dict['causal_struct'] = np.array(G_und.get_adjacency().data).T
        self.data_dict['signed_causal_struct'] = None

    def generate_dag_graph(self, p_edge=0.3, signed=True):
        """
        Generate a random Directed Acyclic Graph (DAG) as the causal structure.
        
        Args:
            p_edge (float): probability of creating an edge between two nodes (0-1).
            signed (bool): if True, assign +1/-1 to edges for activation/inhibition.
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # Random topological order to enforce acyclicity
        ordering = np.random.permutation(self.num_vars)
        
        # Initialize adjacency
        adj = np.zeros((self.num_vars, self.num_vars), dtype=int)

        # Build edges only from earlier -> later in ordering
        for i in range(self.num_vars):
            for j in range(i+1, self.num_vars):
                if np.random.rand() < p_edge:
                    src = ordering[i]
                    dst = ordering[j]
                    adj[src, dst] = 1

        # Store unsigned adjacency (causal structure)
        self.data_dict['causal_struct'] = adj

        # Optional: assign +1/-1 for causal influence
        if signed:
            signed_adj = adj.copy()
            signs = np.random.choice([-1, 1], size=signed_adj.shape, replace=True)
            signed_adj = signed_adj * signs
            self.data_dict['signed_causal_struct'] = signed_adj
        else:
            self.data_dict['signed_causal_struct'] = None


    def generate_example(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        x_n_list = []
        x_ab_list = []
        eps_n_list = []
        eps_ab_list = []
        label_list = []

        coefficients = np.random.uniform(low=0.1, high=2.0, size=(self.num_vars, self.num_vars, 5))

        for i in tqdm(range(self.n)):
            # Generate noise based on dependency flag.
            if self.dependent_features == 1:
                # Generate features with specified covariance
                # Define a covariance matrix manually
                covariance_matrix = np.array([
                    [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
                    [0.8, 1.0, 0.7, 0.5, 0.3, 0.2],
                    [0.6, 0.7, 1.0, 0.6, 0.4, 0.3],
                    [0.4, 0.5, 0.6, 1.0, 0.5, 0.4],
                    [0.2, 0.3, 0.4, 0.5, 1.0, 0.6],
                    [0.1, 0.2, 0.3, 0.4, 0.6, 1.0]
                ])
                mean = np.zeros(self.num_vars)
                eps = self.noise_scale * np.random.multivariate_normal(mean, covariance_matrix, size=self.t)
            else:
                eps = self.noise_scale * np.random.randn(self.t, self.num_vars)

            # Make separate copies for normal and anomalous series.
            eps_normal = eps.copy()
            eps_anom = eps.copy()

            # Initialize time series arrays with random initial values for the first 5 time steps.
            x = np.zeros((self.t, self.num_vars))
            x[:5] = np.random.randn(5, self.num_vars)
            x_ab = np.zeros((self.t, self.num_vars))
            x_ab[:5] = x[:5].copy()

            A_list = [self.data_dict['causal_struct'] * coefficients[:, :, lag] for lag in range(5)]

            # Set up anomaly parameters.
            t_p = np.random.randint(int(0.2 * self.t), int(0.8 * self.t), size=1)
            if self.adlength > 1:
                t_p = np.array([t_p[0] + j for j in range(self.adlength)])
            t_p_set = set(t_p)  # Use a set for O(1) membership checking.
            feature_count = np.random.randint(1, min(10, self.num_vars) + 1)
            feature_p = np.random.permutation(np.arange(self.num_vars))[:feature_count]
            ab = np.zeros(self.num_vars)
            ab[feature_p] += self.mul
            temp_label = np.zeros((self.t, self.num_vars))
            temp_label[t_p, feature_p] = 1

            # Generate the normal time series x using the vectorized inner loop.
            for t in range(5, self.t):
                # Sum contributions from the previous 5 time steps.
                #contributions = sum(
                #    A_list[lag].dot(np.cos(x[t - lag - 1, :] + 1)) for lag in range(5)
                #)
                #contributions = sum(
                #    A_list[lag].dot(
                #        np.sin(x[t - lag - 1, :] + 0.5) + np.tanh(x[t - lag - 1, :]**2) + np.log1p(np.abs(x[t - lag - 1, :]))
                #    )
                #    for lag in range(5)
                #)
                contributions = sum(
                    A_list[lag].dot(
                        np.sin(x[t - lag - 1, :] + 0.5) + np.log1p(np.abs(x[t - lag - 1, :]))
                    )
                    for lag in range(5)
                )
                x[t, :] = contributions + eps_normal[t, :]

            # Generate the anomalous time series x_ab.
            ab_effect = np.zeros(self.num_vars)  # For causal adtype smoothing
            for t in range(5, self.t):
                # For anomaly time steps, update the noise with the anomaly effect.
                if t in t_p_set:
                    if self.adtype == 'non_causal':
                        eps_anom[t, :] += ab
                    elif self.adtype == 'causal':
                        ab_effect = 0.95 * ab_effect + ab  # decay factor <1 for smoothing
                    else:
                        raise NotImplementedError("Invalid adtype. Expected 'non_causal' or 'causal'.")
                # Apply causal effect at every timestep (decay continues even outside t_p_set)
                if self.adtype == 'causal':
                    eps_anom[t, :] += ab_effect
                    ab_effect = 0.95 * ab_effect  # decay for next timestep
                contributions_ab = sum(
                    A_list[lag].dot(np.cos(x_ab[t - lag - 1, :] + 1)) for lag in range(5)
                )
                x_ab[t, :] = contributions_ab + eps_anom[t, :]

            x_n_list.append(x)
            eps_n_list.append(eps_normal)
            x_ab_list.append(x_ab)
            eps_ab_list.append(eps_anom)
            label_list.append(temp_label)

        # Save the generated lists into the data dictionary (done once after the loop for efficiency).
        self.data_dict['x_n_list'] = np.array(x_n_list)
        self.data_dict['x_ab_list'] = np.array(x_ab_list)
        self.data_dict['eps_n_list'] = np.array(eps_n_list)
        self.data_dict['eps_ab_list'] = np.array(eps_ab_list)
        self.data_dict['label_list'] = np.array(label_list)

    def generate_example_(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        x_n_list = []
        x_ab_list = []
        eps_n_list = []
        eps_ab_list = []
        label_list = []

        # Random coefficients for each lag
        coefficients = np.random.uniform(low=0.1, high=2.0, size=(self.num_vars, self.num_vars, 5))

        for i in tqdm(range(self.n)):

            # -------------------------
            # Noise generation
            # -------------------------
            if self.dependent_features == 1:
                covariance_matrix = np.array([
                    [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
                    [0.8, 1.0, 0.7, 0.5, 0.3, 0.2],
                    [0.6, 0.7, 1.0, 0.6, 0.4, 0.3],
                    [0.4, 0.5, 0.6, 1.0, 0.5, 0.4],
                    [0.2, 0.3, 0.4, 0.5, 1.0, 0.6],
                    [0.1, 0.2, 0.3, 0.4, 0.6, 1.0]
                ])
                mean = np.zeros(self.num_vars)
                eps = self.noise_scale * np.random.multivariate_normal(mean, covariance_matrix, size=self.t)
            else:
                eps = self.noise_scale * np.random.randn(self.t, self.num_vars)

            eps_normal = eps.copy()
            eps_anom = eps.copy()

            # -------------------------
            # Initialization
            # -------------------------
            x = np.zeros((self.t, self.num_vars))
            x[:5] = np.random.randn(5, self.num_vars)

            x_ab = np.zeros((self.t, self.num_vars))
            x_ab[:5] = x[:5].copy()

            A_list = [self.data_dict['causal_struct'] * coefficients[:, :, lag] for lag in range(5)]

            # -------------------------
            # Anomaly parameters
            # -------------------------
            t_p = np.random.randint(int(0.2 * self.t), int(0.8 * self.t), size=1)
            if self.adlength > 1:
                t_p = np.array([t_p[0] + j for j in range(self.adlength)])

            t_p_set = set(t_p)

            feature_count = np.random.randint(1, min(10, self.num_vars) + 1)
            feature_p = np.random.permutation(np.arange(self.num_vars))[:feature_count]

            ab = np.zeros(self.num_vars)
            ab[feature_p] += self.mul

            # -------------------------
            # Labels (with propagation)
            # -------------------------
            temp_label = np.zeros((self.t, self.num_vars))

            # FIX-3: Mark anomaly time AND next time step (propagation)
            for tp in t_p:
                temp_label[tp, feature_p] = 1
                if tp + 1 < self.t:
                    temp_label[tp + 1, feature_p] = 1    # first-level propagation

            # -------------------------
            # Generate normal series
            # -------------------------
            for t in range(5, self.t):
                contributions = np.zeros(self.num_vars)
                for lag in range(5):
                    prev = x[t - lag - 1, :]

                    nonlinear_prev = (
                        np.cos(prev + 1) +
                        0.5 * np.sin(prev)
                    )

                    contributions += A_list[lag].dot(nonlinear_prev)

                x[t, :] = contributions + eps_normal[t, :]

            # -------------------------
            # Generate anomalous series
            # -------------------------
            for t in range(5, self.t):

                # FIX-1: Inject anomaly into state (persistent + causal propagation)
                if t in t_p_set:
                    x_ab[t - 1, feature_p] += self.mul

                contributions_ab = np.zeros(self.num_vars)
                for lag in range(5):
                    prev_ab = x_ab[t - lag - 1, :]

                    nonlinear_prev_ab = (
                        np.cos(prev_ab + 1) +
                        0.5 * np.sin(prev_ab)
                    )

                    # FIX-2: Stronger causal propagation
                    contributions_ab += 1.3 * A_list[lag].dot(nonlinear_prev_ab)

                x_ab[t, :] = contributions_ab + eps_anom[t, :]

            # Append all outputs
            x_n_list.append(x)
            eps_n_list.append(eps_normal)
            x_ab_list.append(x_ab)
            eps_ab_list.append(eps_anom)
            label_list.append(temp_label)

        # Save in dict
        self.data_dict['x_n_list'] = np.array(x_n_list)
        self.data_dict['x_ab_list'] = np.array(x_ab_list)
        self.data_dict['eps_n_list'] = np.array(eps_n_list)
        self.data_dict['eps_ab_list'] = np.array(eps_ab_list)
        self.data_dict['label_list'] = np.array(label_list)


    def generate_example2(self):
        """
        Generate paired normal and anomalous time series with causal propagation of anomalies.
        - Inject anomaly into state (x_ab) at t-1 for root features.
        - Propagate anomaly to children for `propagation_depth` steps (BFS along causal_struct).
        - Scale anomaly magnitude by node out-degree (so hubs propagate stronger signals).
        - Produce labels that mark all affected nodes at each propagated timestep.
        """

        # Defaults (can be set on the object before calling)
        propagation_depth = getattr(self, "propagation_depth", 3)        # how many steps forward to propagate
        propagation_scale_factor = getattr(self, "propagation_scale_factor", 1.0)  # global multiplier for anomaly strength
        base_propagation_damp = getattr(self, "propagation_damp", 0.7)  # damp factor per depth (divides magnitude each depth)

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        x_n_list = []
        x_ab_list = []
        eps_n_list = []
        eps_ab_list = []
        label_list = []

        # Random coefficients for each lag
        coefficients = np.random.uniform(low=0.1, high=2.0, size=(self.num_vars, self.num_vars, 5))

        # helper: get children (assuming causal_struct[f, c] != 0 means f -> c)
        causal_struct = np.array(self.data_dict.get('causal_struct', np.zeros((self.num_vars, self.num_vars))))

        def bfs_propagation(seed_nodes, max_depth):
            """
            Returns a list-of-lists: nodes_by_depth[0] == seed_nodes,
            nodes_by_depth[1] == children of seed_nodes, etc. Duplicates removed per depth.
            """
            nodes_by_depth = []
            visited = set(seed_nodes)
            current = list(seed_nodes)
            nodes_by_depth.append(current)

            for d in range(1, max_depth + 1):
                next_nodes = []
                for n in current:
                    children = np.where(causal_struct[n, :] != 0)[0].tolist()
                    for c in children:
                        if c not in visited:
                            next_nodes.append(c)
                            visited.add(c)
                if len(next_nodes) == 0:
                    break
                nodes_by_depth.append(next_nodes)
                current = next_nodes

            return nodes_by_depth

        for i in tqdm(range(self.n)):

            # -------------------------
            # Noise generation
            # -------------------------
            if self.dependent_features == 1:
                covariance_matrix = np.array([
                    [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
                    [0.8, 1.0, 0.7, 0.5, 0.3, 0.2],
                    [0.6, 0.7, 1.0, 0.6, 0.4, 0.3],
                    [0.4, 0.5, 0.6, 1.0, 0.5, 0.4],
                    [0.2, 0.3, 0.4, 0.5, 1.0, 0.6],
                    [0.1, 0.2, 0.3, 0.4, 0.6, 1.0]
                ])
                mean = np.zeros(self.num_vars)
                eps = self.noise_scale * np.random.multivariate_normal(mean, covariance_matrix, size=self.t)
            else:
                eps = self.noise_scale * np.random.randn(self.t, self.num_vars)

            eps_normal = eps.copy()
            eps_anom = eps.copy()

            # -------------------------
            # Initialization
            # -------------------------
            x = np.zeros((self.t, self.num_vars))
            x[:5] = np.random.randn(5, self.num_vars)

            x_ab = np.zeros((self.t, self.num_vars))
            x_ab[:5] = x[:5].copy()

            A_list = [causal_struct * coefficients[:, :, lag] for lag in range(5)]

            # -------------------------
            # Anomaly parameters
            # -------------------------
            t_p = np.random.randint(int(0.2 * self.t), int(0.8 * self.t), size=1)
            if self.adlength > 1:
                t_p = np.array([t_p[0] + j for j in range(self.adlength)])
            t_p_set = set(t_p)

            feature_count = np.random.randint(1, min(10, self.num_vars) + 1)
            feature_p = np.random.permutation(np.arange(self.num_vars))[:feature_count]

            # base anomaly vector for root features
            ab = np.zeros(self.num_vars)
            ab[feature_p] += self.mul

            # -------------------------
            # Labels (with propagation across causal graph)
            # -------------------------
            temp_label = np.zeros((self.t, self.num_vars), dtype=int)

            # Precompute full propagation chains (nodes_by_depth) for this anomaly event
            # We'll do this per root t_p and root features to allow multiple anomaly starts.
            propagation_chains = {}  # key: tp -> nodes_by_depth (list of lists)
            for tp in t_p:
                nodes_by_depth = bfs_propagation(feature_p, propagation_depth)
                propagation_chains[int(tp)] = nodes_by_depth
                # Mark labels for each depth
                for depth, nodes in enumerate(nodes_by_depth):
                    target_t = int(tp) + depth
                    if target_t >= self.t:
                        break
                    temp_label[target_t, nodes] = 1

            # -------------------------
            # Generate normal time series
            # -------------------------
            for t in range(5, self.t):
                contributions = np.zeros(self.num_vars)
                for lag in range(5):
                    prev = x[t - lag - 1, :]

                    nonlinear_prev = (
                        np.cos(prev + 1) +
                        0.5 * np.sin(prev)
                    )

                    contributions += A_list[lag].dot(nonlinear_prev)

                x[t, :] = contributions + eps_normal[t, :]

            # -------------------------
            # Generate anomalous series with causal propagation applied to state
            # -------------------------
            for t in range(5, self.t):

                # If this timestep is an anomaly start, inject into x_ab prior states for root features
                if t in t_p_set:
                    # Inject root anomaly into previous state so causal dynamics can propagate it
                    # scale by node out-degree so hubs have stronger impact
                    for f in feature_p:
                        out_deg = np.count_nonzero(causal_struct[f, :])
                        degree_scale = 1.0 + (out_deg / max(1, self.num_vars))  # mild scale by out-degree
                        x_ab[t - 1, f] += self.mul * propagation_scale_factor * degree_scale

                    # Inject propagated anomalies into descendant states for multiple depths
                    # nodes_by_depth[0] are root features; we already injected them at t-1
                    nodes_by_depth = propagation_chains[int(t)]
                    for depth, nodes in enumerate(nodes_by_depth):
                        if depth == 0:
                            continue  # root already handled
                        target_index = t - 1 + depth  # place into state such that it will be used in subsequent calculations
                        if target_index < 0 or target_index >= self.t:
                            continue
                        # damp magnitude with depth so deeper nodes get smaller direct injection
                        damp = (base_propagation_damp ** (depth - 1))
                        for node in nodes:
                            # scale by out-degree of the source parents to make propagation realistic
                            parent_deg = np.count_nonzero(causal_struct[:, node])
                            parent_scale = 1.0 + (parent_deg / max(1, self.num_vars))
                            added = self.mul * propagation_scale_factor * damp * parent_scale / (depth + 1)
                            x_ab[target_index, node] += added

                # Additionally, if you prefer injecting some anomaly into noise for immediate spike:
                # (optional) we can boost eps_anom at anomaly timesteps for root features
                if t in t_p_set:
                    eps_anom[t, feature_p] += self.mul * 0.2 * propagation_scale_factor

                # Now compute contributions for this timestep using current x_ab (which may contain injected anomalies)
                contributions_ab = np.zeros(self.num_vars)
                for lag in range(5):
                    prev_ab = x_ab[t - lag - 1, :]

                    nonlinear_prev_ab = (
                        np.cos(prev_ab + 1) +
                        0.5 * np.sin(prev_ab)
                    )

                    # keep a slightly stronger causal effect to help propagation be visible
                    contributions_ab += 1.3 * A_list[lag].dot(nonlinear_prev_ab)

                x_ab[t, :] = contributions_ab + eps_anom[t, :]

            # Append outputs
            x_n_list.append(x)
            eps_n_list.append(eps_normal)
            x_ab_list.append(x_ab)
            eps_ab_list.append(eps_anom)
            label_list.append(temp_label)

        # Save in dict (as numpy arrays)
        self.data_dict['x_n_list'] = np.array(x_n_list)
        self.data_dict['x_ab_list'] = np.array(x_ab_list)
        self.data_dict['eps_n_list'] = np.array(eps_n_list)
        self.data_dict['eps_ab_list'] = np.array(eps_ab_list)
        self.data_dict['label_list'] = np.array(label_list)


    def save_data(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        np.save(os.path.join(self.data_dir, 'x_n_list'), self.data_dict['x_n_list'])
        np.save(os.path.join(self.data_dir, 'x_ab_list'), self.data_dict['x_ab_list'])
        np.save(os.path.join(self.data_dir, 'eps_n_list'), self.data_dict['eps_n_list'])
        np.save(os.path.join(self.data_dir, 'eps_ab_list'), self.data_dict['eps_ab_list'])
        np.save(os.path.join(self.data_dir, 'causal_struct'), self.data_dict['causal_struct'])
        np.save(os.path.join(self.data_dir, 'label_list'), self.data_dict['label_list'])

    def load_data(self):
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'))
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'))
        self.data_dict['eps_n_list'] = np.load(os.path.join(self.data_dir, 'eps_n_list.npy'))
        self.data_dict['eps_ab_list'] = np.load(os.path.join(self.data_dir, 'eps_ab_list.npy'))
        self.data_dict['causal_struct'] = np.load(os.path.join(self.data_dir, 'causal_struct.npy'))
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'))
        self.data_dict['signed_causal_struct'] = None
