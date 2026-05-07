from collections import defaultdict

import mo_gymnasium as mo_gym
import numpy as np
from scipy.spatial import ConvexHull
from tqdm import tqdm


class ResourceGatheringVI:
    def __init__(
        self,
        gamma=0.99,
        theta=1e-5,
        env_name="resource-gathering-v0",
        init_state_dist="start",
    ):
        self.gamma = gamma
        self.theta = theta
        self.env = mo_gym.make(env_name)

                                            
        self.n_actions = self.env.action_space.n
                                                  
        self.n_objectives = 3

        self._discover_states()
        self._build_stochastic_model()
        self.set_initial_distribution(init_state_dist)

    def _set_env_state(self, state):
        """Safely forces the environment into a specific state."""
        self.env.reset()
        unwrapped = self.env.unwrapped

                                                                  
        unwrapped.current_pos = np.array([state[0], state[1]])
        unwrapped.has_gold = int(state[2])
        unwrapped.has_gem = int(state[3])

    def _discover_states(self):
        """Generates all valid states plus ONE absorbing terminal state."""
        self.states = []
        self.state_to_idx = {}

                                               
        for r in range(5):
            for c in range(5):
                for g in range(2):
                    for d in range(2):
                        s = (r, c, g, d)
                        self.state_to_idx[s] = len(self.states)
                        self.states.append(s)

                                                                                  
        self.START_STATE = (4, 2, 0, 0)

                                              
        self.TERMINAL_STATE = (-1, -1, -1, -1)
        self.state_to_idx[self.TERMINAL_STATE] = len(self.states)
        self.states.append(self.TERMINAL_STATE)

        self.n_states = len(self.states)
        self.terminal_idx = self.state_to_idx[self.TERMINAL_STATE]

    def _build_stochastic_model(self):
        """
        Inflated 3D Model WITH WIND: 20% chance to slip and execute a random action.
        This forces the DAgger state-coverage failure in Mixed BC datasets.
        """
        print("Building Inflated White-Box transition model WITH WIND...")
        self.P_tensor = np.zeros(
            (self.n_states, self.n_actions, self.n_states), dtype=np.float32
        )
        self.R_tensor = np.zeros(
            (self.n_states, self.n_actions, self.n_states, self.n_objectives),
            dtype=np.float32,
        )

                                           
        HOME = (4, 2)                 
        GOLD_POS = (0, 2)              
        GEM_POS = (1, 4)                           
        ENEMIES = [(0, 3), (1, 2)]                           

                                                                        
        slip_prob = 0.05

        for s_idx, state in enumerate(self.states):
            if state == self.TERMINAL_STATE:
                for a in range(self.n_actions):
                    self.P_tensor[s_idx, a, s_idx] = 1.0
                continue

            r, c, has_gold, has_gem = state

            for a in range(self.n_actions):
                                                                  
                action_probs = np.zeros(self.n_actions)
                for executed_a in range(self.n_actions):
                    if executed_a == a:
                                                                                                  
                        action_probs[executed_a] = (1.0 - slip_prob) + (
                            slip_prob / self.n_actions
                        )
                    else:
                                                                    
                        action_probs[executed_a] = slip_prob / self.n_actions

                                                                                                     
                temp_P = defaultdict(float)
                temp_R = defaultdict(lambda: np.zeros(3, dtype=np.float32))

                                                           
                for exec_a, prob in enumerate(action_probs):
                    if prob == 0:
                        continue

                    nr, nc = r, c
                    if exec_a == 0:
                        nr -= 1      
                    elif exec_a == 1:
                        nr += 1        
                    elif exec_a == 2:
                        nc -= 1        
                    elif exec_a == 3:
                        nc += 1         

                    nr, nc = max(0, min(4, nr)), max(0, min(4, nc))

                                  
                    ngold = 1 if has_gold == 1 or (nr, nc) == GOLD_POS else 0
                    ngem = 1 if has_gem == 1 or (nr, nc) == GEM_POS else 0

                    is_enemy = (nr, nc) in ENEMIES

                                                                                                          
                    is_home = ((nr, nc) == HOME) and (ngold == 1 or ngem == 1)

                    if is_enemy:
                                                       
                        base_death_prob = 0.1                
                        gem_penalty_multiplier = 2.5
                        death_prob = base_death_prob * (
                            gem_penalty_multiplier if has_gem else 1.0
                        )

                                       
                        temp_P[self.terminal_idx] += prob * death_prob
                        temp_R[self.terminal_idx] += (prob * death_prob) * np.array(
                            [-1.0, 0.0, 0.0]
                        )

                                          
                        safe_state = (nr, nc, ngold, ngem)
                        safe_idx = self.state_to_idx[safe_state]
                        temp_P[safe_idx] += prob * (1.0 - death_prob)

                    elif is_home:
                                                               
                        gold_val = 1.0
                        gem_val = 2.0

                        temp_P[self.terminal_idx] += prob
                        temp_R[self.terminal_idx] += prob * np.array(
                            [0.0, float(ngold) * gold_val, float(ngem) * gem_val]
                        )

                    else:
                                         
                        next_state = (nr, nc, ngold, ngem)
                        ns_idx = self.state_to_idx[next_state]
                        temp_P[ns_idx] += prob

                                                                     
                for n_s, p_val in temp_P.items():
                    self.P_tensor[s_idx, a, n_s] = p_val
                    if p_val > 0:
                                                                                      
                        self.R_tensor[s_idx, a, n_s, :] = temp_R[n_s] / p_val

    def set_initial_distribution(self, dist_config):
        """
        Configures the initial state distribution (\rho).
        - "start": 100% at the normal home spawn (Default)
        - "uniform": Uniform over all valid playable states
        - float (0.0 to 1.0): Alpha-mixture between "start" and "uniform"
        """
        self.rho = np.zeros(self.n_states, dtype=np.float32)

                                                                                               
        valid_indices = []
        for s_idx, state in enumerate(self.states):
            if state == self.TERMINAL_STATE:
                continue
                                                                                              
            if (
                state[0] == self.START_STATE[0]
                and state[1] == self.START_STATE[1]
                and (state[2] == 1 or state[3] == 1)
            ):
                continue
            valid_indices.append(s_idx)

        num_valid = len(valid_indices)

        if isinstance(dist_config, str):
            if dist_config == "start":
                self.rho[self.state_to_idx[self.START_STATE]] = 1.0
            elif dist_config == "uniform":
                for idx in valid_indices:
                    self.rho[idx] = 1.0 / num_valid
            else:
                raise ValueError(f"Unknown dist_config string: {dist_config}")

        elif isinstance(dist_config, float):
                                                                  
            alpha = np.clip(dist_config, 0.0, 1.0)
            for idx in valid_indices:
                self.rho[idx] = alpha / num_valid
            self.rho[self.state_to_idx[self.START_STATE]] += 1.0 - alpha

        else:
            raise ValueError(
                "Invalid dist_config format. Use 'start', 'uniform', or a float."
            )

    def reset_random(self, start_state=None):
        """
        Resets the environment according to the configured \rho distribution,
        or a specific state if provided.
        """
        self.env.reset()

        if start_state is None:
                                                                                      
            idx = np.random.choice(self.n_states, p=self.rho)
            state = self.states[idx]
        else:
            state = start_state

                                                                                      
        unwrapped = self.env.unwrapped
        unwrapped.current_pos = np.array([state[0], state[1]])
        unwrapped.has_gold = int(state[2])
        unwrapped.has_gem = int(state[3])

                                                                          
        obs = np.array(state, dtype=np.int32)

        return obs, {}

    def solve(self, weight):
        weight = np.array(weight, dtype=np.float32)

        scalar_R = self.R_tensor @ weight
        expected_R = np.sum(self.P_tensor * scalar_R, axis=-1)

        V = np.zeros(self.n_states, dtype=np.float32)

        for _ in range(1000):
            Q = expected_R + self.gamma * np.sum(
                self.P_tensor * V[None, None, :], axis=-1
            )
            new_V = np.max(Q, axis=-1)

                                        
            new_V[self.terminal_idx] = 0.0

            delta = np.max(np.abs(V - new_V))
            V = new_V
            if delta < self.theta:
                break

        Q = expected_R + self.gamma * np.sum(self.P_tensor * V[None, None, :], axis=-1)
        expected_obj0 = np.sum(self.P_tensor * self.R_tensor[..., 0], axis=-1)
        policy_array = np.argmax(Q + 1e-9 * expected_obj0, axis=-1)

        return {s: policy_array[i] for i, s in enumerate(self.states)}

    def get_average_return(self, policy):
        """Policy evaluation for Stochastic MDPs (No masking required!)."""
        R_pi = np.zeros((self.n_states, self.n_objectives))
        P_pi = np.zeros((self.n_states, self.n_states))

        for s_idx, state in enumerate(self.states):
            a = policy[state]

                                                                 
            R_pi[s_idx] = np.sum(
                np.expand_dims(self.P_tensor[s_idx, a, :], -1)
                * self.R_tensor[s_idx, a, :, :],
                axis=0,
            )

                                                                         
            P_pi[s_idx, :] = self.P_tensor[s_idx, a, :]

        A = np.eye(self.n_states) - self.gamma * P_pi
        V = np.linalg.solve(A, R_pi)

        ret = np.dot(self.rho, V)
        return tuple(float(x) for x in ret)

    def _get_policy_matrices(self, policy):
        """Helper for N-Dimensional Updates (No masking required!)."""
        R_pi = np.zeros((self.n_states, self.n_objectives))
        P_pi = np.zeros((self.n_states, self.n_states))

        for s_idx, state in enumerate(self.states):
            a = policy[state]

            R_pi[s_idx] = np.sum(
                np.expand_dims(self.P_tensor[s_idx, a, :], -1)
                * self.R_tensor[s_idx, a, :, :],
                axis=0,
            )
            P_pi[s_idx, :] = self.P_tensor[s_idx, a, :]

        return P_pi, R_pi

    def find_ccs_ols(self, tolerance=1e-5):
        """
        N-Dimensional Optimistic Linear Support (OLS).
        """
        print("\nRunning 3D Optimistic Linear Support (OLS)...")

        extremes = np.eye(self.n_objectives)
        ccs_policies = {}
        ccs_returns = []

        for w in extremes:
            pol = self.solve(w)
            ret = self.get_average_return(pol)
            if not any(
                np.allclose(ret, existing, atol=tolerance) for existing in ccs_returns
            ):
                ccs_returns.append(ret)
                ccs_policies[ret] = pol

        if len(ccs_returns) <= self.n_objectives:
            w_uniform = np.ones(self.n_objectives) / self.n_objectives
            pol = self.solve(w_uniform)
            ret = self.get_average_return(pol)
            if not any(
                np.allclose(ret, existing, atol=tolerance) for existing in ccs_returns
            ):
                ccs_returns.append(ret)
                ccs_policies[ret] = pol

        processed_weights = []

        while True:
            points = np.array(ccs_returns)

            try:
                hull = ConvexHull(points)
            except Exception:
                print(
                    "\n[Geometry Check] Hull volume collapsed. The 3D Pareto front is flat or fully discovered."
                )
                break

            new_points_found = False

            for eq in hull.equations:
                w = eq[:-1]
                w_sum = np.sum(np.abs(w))
                if w_sum < 1e-9:
                    continue
                w_norm = np.abs(w / w_sum)

                if any(np.allclose(w_norm, pw, atol=1e-3) for pw in processed_weights):
                    continue

                processed_weights.append(w_norm)

                pol_new = self.solve(w_norm)
                ret_new = self.get_average_return(pol_new)

                current_max_val = max(np.dot(w_norm, p) for p in points)
                new_val = np.dot(w_norm, ret_new)

                if new_val > current_max_val + tolerance:
                    if not any(
                        np.allclose(ret_new, existing, atol=tolerance)
                        for existing in ccs_returns
                    ):
                        ccs_returns.append(ret_new)
                        ccs_policies[ret_new] = pol_new
                        new_points_found = True
                        print(
                            f"OLS Discovered new vertex: {ret_new} via weight {np.round(w_norm, 2)}"
                        )

            if not new_points_found:
                break

        print(f"OLS Complete. Exact Convex Coverage Set size: {len(ccs_returns)}")
        return [(np.array(r), p) for r, p in ccs_policies.items()]

    def find_exact_pareto_front(self):
        """Phase 1: Deterministic Pareto Front upgraded for N-Dimensional stochasticities."""

        def hash_policy(p):
            return tuple(p[s] for s in self.states)

        initial_policy = self.solve([1.0, 0.0, 0.0])
        initial_ret_t = self.get_average_return(initial_policy)

        found_returns = {initial_ret_t: [initial_policy]}
        queue = [initial_policy]
        processed_policies = {hash_policy(initial_policy)}

        print("\nSearching Exact Deterministic Pareto Front (N-D)...")
        pbar = tqdm(total=1)

        processed_count = 0
        while processed_count < len(queue):
            curr_policy = queue[processed_count]
            processed_count += 1
            curr_ret_t = self.get_average_return(curr_policy)

            if curr_ret_t not in found_returns:
                continue

            P_curr, R_curr = self._get_policy_matrices(curr_policy)
            A_curr = np.eye(self.n_states) - self.gamma * P_curr
            A_inv = np.linalg.inv(A_curr)
            V_curr = A_inv @ R_curr
            mu_T_A_inv = self.rho @ A_inv

            for idx, state in enumerate(self.states):
                curr_action = curr_policy[state]

                for action in range(self.n_actions):
                    if action == curr_action:
                        continue

                    n_pol = curr_policy.copy()
                    n_pol[state] = action

                    n_hash = hash_policy(n_pol)
                    if n_hash in processed_policies:
                        continue
                    processed_policies.add(n_hash)

                                                              
                    R_new_i = np.sum(
                        np.expand_dims(self.P_tensor[idx, action, :], -1)
                        * self.R_tensor[idx, action, :, :],
                        axis=0,
                    )

                                                              
                    P_new_i = self.P_tensor[idx, action, :]

                    delta_P_i = P_new_i - P_curr[idx]
                    delta_R_i = R_new_i - R_curr[idx]

                    v = self.gamma * delta_P_i
                    val = delta_R_i + self.gamma * np.dot(delta_P_i, V_curr)
                    c_s = A_inv[:, idx]

                    denominator = 1 - np.dot(v, c_s)

                    if abs(denominator) < 1e-9:
                        n_ret_t = self.get_average_return(n_pol)
                    else:
                        delta_J = (val / denominator) * mu_T_A_inv[idx]
                        n_ret_t = tuple(curr_ret_t + delta_J)

                                                         
                    n_ret_ary = np.array(n_ret_t)
                    is_dominated = False
                    for ex in list(found_returns.keys()):
                        ex_ary = np.array(ex)
                        if np.all(ex_ary >= n_ret_ary) and np.any(ex_ary > n_ret_ary):
                            is_dominated = True
                            break

                    if is_dominated:
                        continue

                    to_remove = []
                    for ex in list(found_returns.keys()):
                        ex_ary = np.array(ex)
                        if np.all(n_ret_ary >= ex_ary) and np.any(n_ret_ary > ex_ary):
                            to_remove.append(ex)

                    for r_t in to_remove:
                        del found_returns[r_t]

                    if n_ret_t not in found_returns:
                        found_returns[n_ret_t] = [n_pol]
                    else:
                        found_returns[n_ret_t].append(n_pol)

                    queue.append(n_pol)

            pbar.total = len(queue)
            pbar.update(1)

        pbar.close()

        results = [(np.array(r_t), pols[0]) for r_t, pols in found_returns.items()]
        return results


if __name__ == "__main__":
    solver = ResourceGatheringVI()
    pareto_front = solver.find_ccs_ols()
    print(f"\nDiscovered {len(pareto_front)} exact deterministic policies!")
