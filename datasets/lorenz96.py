import numpy as np
from scipy.integrate import ode
from tqdm import tqdm
import os

class Lorenz96:
    def __init__(self, options):
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.n = options['training_size'] + options['testing_size']
        self.t = options['T']
        self.num_vars = options['num_vars']
        self.data_dir = options['data_dir']
        self.mul = options['mul']
        self.adlength = options['adlength']
        self.adtype = options['adtype']
        self.downsample_factor = options['downsample_factor']
        self.system_ode = ode(self._system_ode).set_integrator('vode', method='bdf')
        self.force = options['force']
        self.dependent_features = options['dependent_features']

    @staticmethod
    def _system_ode(t: float, x: np.array, theta: np.array) -> list:
        """
        Computes the Lorenz96 derivatives using vectorized operations.
        """
        force = theta[1]
        # Flatten the state vector and determine its dimension
        x = np.asarray(x).flatten()
        # Compute the derivative using vectorized cyclic shifts.
        f = (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + force
        return f.tolist()

    def gen_data_point__(self, downsample=True):
        """
        Integrate the Lorenz96 system using an ODE solver and inject anomalies.
        Downsampling and anomaly injection are handled as in the original code,
        but with improvements in accumulation and anomaly detection.
        """
        # Integration parameters
        t_start = 0.0
        t_delta_integration = 0.01
        t_end = 10 * (self.t - 1) * t_delta_integration

        # Generate initial states
        x = list(np.random.rand(self.num_vars, 1))
        # other
        x_n = np.copy(x).reshape(self.num_vars, 1)
        t = [t_start]
        # initialize LV ODE system
        self.system_ode.set_initial_value(x, t_start).set_f_params([self.num_vars, self.force])
        # integrate until specified
        while self.system_ode.successful() and self.system_ode.t < t_end:
            self.system_ode.integrate(t=self.system_ode.t + t_delta_integration)
            x_n = np.c_[x_n, self.system_ode.y.reshape(self.num_vars, 1)]
            t.append(self.system_ode.t)
        # Downsample the time series
        if downsample:
            x_n = x_n[:, ::self.downsample_factor]
        x_ab = np.copy(x)
        lst_label = np.zeros((self.t*self.downsample_factor, self.num_vars))
        t_ab = np.random.randint(int(0.5*self.t), int(0.8*self.t), size=1)
        if self.adlength > 1:
            temp_t_ab = []
            for i in range(self.adlength):
                temp_t_ab.append(t_ab+i)
            t_ab = np.array(temp_t_ab)
        feature_ab = np.random.permutation(np.arange(self.num_vars))[:np.random.randint(1, self.num_vars//2+1)]
        #lst_label[t_ab*self.downsample_factor, feature_ab] = 1
        ode_steps_per_sample = int(1 / t_delta_integration)
        anomaly_steps = t_ab * ode_steps_per_sample
        lst_label[anomaly_steps, feature_ab] = 1

        self.system_ode.set_initial_value(x_ab, t_start).set_f_params([self.num_vars, self.force])
        # integrate until specified
        while self.system_ode.successful() and self.system_ode.t < t_end:
            self.system_ode.integrate(t=self.system_ode.t + t_delta_integration)
            #if int((1/t_delta_integration)*(self.system_ode.t)) in t_ab*self.downsample_factor:
            current_step = int(self.system_ode.t / t_delta_integration)
            if current_step in anomaly_steps:
                ##if self.adtype == 'non_causal':
                ##    assert self.system_ode.y.reshape(-1,1).shape == lst_label[int(self.system_ode.t), :].reshape(-1,1).shape
                ##    x_ab = np.c_[x_ab, self.system_ode.y.reshape(self.num_vars, 1) + self.mul *
                ##                       np.array((lst_label[int((1/t_delta_integration)*(self.system_ode.t)), :]).reshape(self.num_vars, 1))]
                ##    self.system_ode.set_initial_value(x_ab[:, -1], self.system_ode.t)
                if self.adtype == 'causal':
                    # Apply anomaly by modifying the ODE internal state (no direct spike)
                    # This causes the anomaly to propagate dynamically through the Lorenz96 system.
                    
                    # 1. Copy current clean state
                    perturbed_state = self.system_ode.y.copy()
                    
                    # 2. Inject perturbation only on the selected features
                    perturbed_state[feature_ab] += self.mul
                    
                    # 3. Restart integrator at the perturbed state
                    self.system_ode.set_initial_value(perturbed_state, self.system_ode.t)
                    
                    # 4. Save the perturbed observation (no explicit spike added)
                    x_ab = np.c_[x_ab, perturbed_state.reshape(self.num_vars, 1)]
                else:
                    raise NotImplementedError("Invalid adtype. Expected 'non_causal' or 'causal'.")
            else:
                x_ab = np.c_[x_ab, self.system_ode.y.reshape(self.num_vars, 1)]
        x_ab = x_ab[:, ::self.downsample_factor]
        lst_label = lst_label[::self.downsample_factor, :]
        assert lst_label.shape == np.transpose(x_n).shape
        assert lst_label.shape == np.transpose(x_ab).shape, print(lst_label.shape, np.transpose(x_ab).shape)
        return np.transpose(x_n), np.transpose(x_ab), lst_label

    def gen_data_point(self, downsample=True):
        """
        Integrate the Lorenz96 system using an ODE solver and inject anomalies.
        Robust to arbitrary integer downsample_factor >= 1.
        """
        # Integration parameters
        t_start = 0.0
        t_delta_integration = 0.01
        # t_end chosen so that there are 10*(self.t-1) integration steps (matches original behavior)
        t_end = 10 * (self.t - 1) * t_delta_integration

        # Generate initial states (column vector per variable)
        x = list(np.random.rand(self.num_vars, 1))
        x_n = np.copy(x).reshape(self.num_vars, 1)
        t = [t_start]

        # initialize ODE system
        self.system_ode.set_initial_value(x, t_start).set_f_params([self.num_vars, self.force])

        # integrate and accumulate raw (no downsampling yet)
        while self.system_ode.successful() and self.system_ode.t < t_end:
            self.system_ode.integrate(t=self.system_ode.t + t_delta_integration)
            x_n = np.c_[x_n, self.system_ode.y.reshape(self.num_vars, 1)]
            t.append(self.system_ode.t)

        # raw length (number of columns) before downsampling
        raw_len = x_n.shape[1]   # typically 1 + 10*(self.t-1)
        # compute steps per "sample" (robust formula)
        if self.t > 1:
            steps_per_sample = max(1, (raw_len - 1) // (self.t - 1))
        else:
            steps_per_sample = 1

        # prepare anomaly label matrix at raw integration-step resolution
        lst_label = np.zeros((raw_len, self.num_vars), dtype=int)

        # sample anomaly sample-indices in the "sample" domain [0..self.t-1]
        # (this preserves your original intent: choose anomaly time between 0.5*t and 0.8*t)
        t_ab = np.random.randint(int(0.5 * self.t), int(0.8 * self.t), size=1)
        if self.adlength > 1:
            temp_t_ab = []
            for i in range(self.adlength):
                temp_t_ab.append(t_ab + i)
            t_ab = np.hstack(temp_t_ab).astype(int)
        else:
            t_ab = t_ab.astype(int)

        # convert sample indices to raw integration-step indices
        anomaly_steps = (t_ab * steps_per_sample).astype(int)

        # choose features to corrupt
        feature_ab = np.random.permutation(np.arange(self.num_vars))[:np.random.randint(1, self.num_vars // 2 + 1)]

        # set labels at raw-step resolution
        # ensure anomaly_steps within bounds
        anomaly_steps = anomaly_steps[anomaly_steps < raw_len]
        if anomaly_steps.size == 0:
            # fallback: if nothing valid (very unlikely), place an anomaly at middle of sequence
            mid = raw_len // 2
            anomaly_steps = np.array([mid])
        lst_label[anomaly_steps, feature_ab] = 1

        # Now build the anomalous trajectory by integrating again, injecting anomalies at raw steps
        x_ab = np.copy(x).reshape(self.num_vars, 1)
        self.system_ode.set_initial_value(x_ab[:, 0], t_start).set_f_params([self.num_vars, self.force])

        # integration counter in raw steps; the ODE's .t increments by t_delta_integration
        while self.system_ode.successful() and self.system_ode.t < t_end:
            self.system_ode.integrate(t=self.system_ode.t + t_delta_integration)
            current_step = int(round(self.system_ode.t / t_delta_integration))

            if current_step in anomaly_steps:
                # label vector at this raw step
                lbl_vec = lst_label[current_step, :].reshape(self.num_vars, 1)

                if self.adtype == 'non_causal':
                    # non-causal: observation shows spike but the system state is NOT restarted from the spiked value
                    disturbed_obs = self.system_ode.y.reshape(self.num_vars, 1) + self.mul * lbl_vec
                    x_ab = np.c_[x_ab, disturbed_obs]
                    # do NOT change the integrator's initial condition -> no propagation
                    # keep integrator state unchanged (it will continue from current true state)
                elif self.adtype == 'causal':
                    # Apply anomaly by modifying the ODE internal state (no direct spike)
                    # This causes the anomaly to propagate dynamically through the Lorenz96 system.
                    
                    # 1. Copy current clean state
                    perturbed_state = self.system_ode.y.copy()
                    
                    # 2. Inject perturbation only on the selected features
                    perturbed_state[feature_ab] += self.mul
                    
                    # 3. Restart integrator at the perturbed state
                    self.system_ode.set_initial_value(perturbed_state, self.system_ode.t)
                    
                    # 4. Save the perturbed observation (no explicit spike added)
                    x_ab = np.c_[x_ab, perturbed_state.reshape(self.num_vars, 1)]
                else:
                    raise NotImplementedError("Invalid adtype. Expected 'non_causal' or 'causal'.")
            else:
                # normal (no anomaly) observation
                x_ab = np.c_[x_ab, self.system_ode.y.reshape(self.num_vars, 1)]

        # Downsample (if requested) both normal and anomalous trajectories and labels
        if downsample and self.downsample_factor > 1:
            x_n_down = x_n[:, ::self.downsample_factor]
            x_ab_down = x_ab[:, ::self.downsample_factor]
            lst_label_down = lst_label[::self.downsample_factor, :]
        else:
            x_n_down = x_n
            x_ab_down = x_ab
            lst_label_down = lst_label

        # final consistency checks: transpose shapes to (time, vars)
        x_n_t = np.transpose(x_n_down)
        x_ab_t = np.transpose(x_ab_down)

        if lst_label_down.shape != x_n_t.shape:
            # If shapes don't match, adapt by trimming/padding labels to match the sequence length.
            # Prefer trimming (most likely cause is small rounding issues); pad with zeros only if needed.
            seq_len = x_n_t.shape[0]
            if lst_label_down.shape[0] > seq_len:
                lst_label_down = lst_label_down[:seq_len, :]
            elif lst_label_down.shape[0] < seq_len:
                pad = np.zeros((seq_len - lst_label_down.shape[0], self.num_vars), dtype=int)
                lst_label_down = np.vstack([lst_label_down, pad])

        assert lst_label_down.shape == x_n_t.shape, (lst_label_down.shape, x_n_t.shape)
        assert lst_label_down.shape == x_ab_t.shape, (lst_label_down.shape, x_ab_t.shape)

        return x_n_t, x_ab_t, lst_label_down

    def get_causal_structure(self):
        """
        Generates a causal structure matrix for the Lorenz96 system.
        """
        a = np.zeros((self.num_vars, self.num_vars))
        for i in range(self.num_vars):
            a[i, i] = 1
            a[(i + 1) % self.num_vars, i] = 1
            a[(i + 2) % self.num_vars, i] = 1
            a[(i - 1) % self.num_vars, i] = 1
        return a

    def generate_example(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        x_n_list = []
        x_ab_list = []
        eps_n_list = []
        eps_ab_list = []
        label_list = []

        for _ in tqdm(range(self.n), desc='Generating data'):
            x_n, x_ab, label = self.gen_data_point()
            x_n_list.append(x_n)
            x_ab_list.append(x_ab)
            eps_n_list.append(np.zeros((self.t, self.num_vars)))
            eps_ab_list.append(np.zeros((self.t, self.num_vars)))
            label_list.append(label)

        self.data_dict['x_n_list'] = np.array(x_n_list)
        self.data_dict['x_ab_list'] = np.array(x_ab_list)
        self.data_dict['eps_n_list'] = np.array(eps_n_list)
        self.data_dict['eps_ab_list'] = np.array(eps_ab_list)
        self.data_dict['label_list'] = np.array(label_list)
        self.data_dict['causal_struct'] = self.get_causal_structure()
        self.data_dict['signed_causal_struct'] = []

    def save_data(self):
        # Create the directory if it does not exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Save the data
        np.save(os.path.join(self.data_dir, 'x_n_list.npy'), self.data_dict['x_n_list'])
        np.save(os.path.join(self.data_dir, 'x_ab_list.npy'), self.data_dict['x_ab_list'])
        np.save(os.path.join(self.data_dir, 'eps_n_list.npy'), self.data_dict['eps_n_list'])
        np.save(os.path.join(self.data_dir, 'eps_ab_list.npy'), self.data_dict['eps_ab_list'])
        np.save(os.path.join(self.data_dir, 'causal_struct.npy'), self.data_dict['causal_struct'])
        np.save(os.path.join(self.data_dir, 'label_list.npy'), self.data_dict['label_list'])


    def load_data(self):
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'))
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'))
        self.data_dict['eps_n_list'] = np.load(os.path.join(self.data_dir, 'eps_n_list.npy'))
        self.data_dict['eps_ab_list'] = np.load(os.path.join(self.data_dir, 'eps_ab_list.npy'))
        self.data_dict['causal_struct'] = np.load(os.path.join(self.data_dir, 'causal_struct.npy'))
        self.data_dict['signed_causal_struct'] = None
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'))

