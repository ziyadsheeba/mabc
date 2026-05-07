import mo_gymnasium as mo_gym
import numpy as np
from tqdm import tqdm


class DSTValueIteration:
    def __init__(
        self,
        gamma=0.99,
        theta=1e-5,
        env_name="deep-sea-treasure-v0",
        init_state_dist="uniform",
    ):
        self.gamma = gamma
        self.theta = theta
        self.env = mo_gym.make(env_name)
        self.map_shape = self.env.unwrapped.sea_map.shape
        self.rows, self.cols = self.map_shape
        self.n_actions = self.env.action_space.n

                                                
        self.transitions = np.zeros(
            (self.rows, self.cols, self.n_actions, 2), dtype=np.int32
        )
        self.rewards = np.zeros(
            (self.rows, self.cols, self.n_actions, 2), dtype=np.float32
        )
        self.terminals = np.zeros((self.rows, self.cols, self.n_actions), dtype=bool)

                                                                           
        self.is_rock = self.env.unwrapped.sea_map == -10
        self.is_treasure = self.env.unwrapped.sea_map > 0
        self.is_terminal_state = self.is_rock | self.is_treasure

                                           
        self.non_terminal_states = []
        self.state_to_idx = {}
        for r in range(self.rows):
            for c in range(self.cols):
                if not self.is_terminal_state[r, c]:
                    self.state_to_idx[(r, c)] = len(self.non_terminal_states)
                    self.non_terminal_states.append((r, c))

        self.n_states = len(self.non_terminal_states)
        self.set_initial_distribution(init_state_dist)
        self._build_model()

    def set_initial_distribution(self, dist_config):
        """
        Configures the initial state distribution (\rho).
        - "start": 100% at (0,0)
        - "uniform": Uniform over all valid states (Default)
        - float (0.0 to 1.0): Alpha-mixture between "start" and "uniform"
        - int (0+): Expanding Manhattan radius from (0,0)
        - dict: Custom mapping of (r,c) -> probability
        """
        self.rho = np.zeros(self.n_states, dtype=np.float32)

        if isinstance(dist_config, str):
            if dist_config == "start":
                self.rho[self.state_to_idx[(0, 0)]] = 1.0
            elif dist_config == "uniform":
                self.rho[:] = 1.0 / self.n_states
            else:
                raise ValueError(f"Unknown dist_config string: {dist_config}")

        elif isinstance(dist_config, float):
                                                                  
            alpha = np.clip(dist_config, 0.0, 1.0)
            self.rho[:] = alpha / self.n_states
            self.rho[self.state_to_idx[(0, 0)]] += 1.0 - alpha

        elif isinstance(dist_config, int):
                                                                         
            valid_states = [
                s for s in self.non_terminal_states if s[0] + s[1] <= dist_config
            ]
            if not valid_states:
                valid_states = [(0, 0)]                         

            prob = 1.0 / len(valid_states)
            for s in valid_states:
                self.rho[self.state_to_idx[s]] = prob

        elif isinstance(dist_config, dict):
                               
            for state, prob in dist_config.items():
                if state in self.state_to_idx:
                    self.rho[self.state_to_idx[state]] = prob
            self.rho /= np.sum(self.rho)             

        else:
            raise ValueError("Invalid dist_config format.")

    def _build_model(self):
        """Builds the deterministic transition and vector reward model."""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.is_terminal_state[r, c]:
                    continue
                for a in range(self.n_actions):
                    self.env.reset()
                    self.env.unwrapped.current_state = np.array([r, c], dtype=np.int32)
                    obs, vec_reward, terminated, truncated, info = self.env.step(a)

                    self.transitions[r, c, a] = obs
                    self.rewards[r, c, a] = vec_reward
                    self.terminals[r, c, a] = terminated

    def solve(self, weight):
        """Solves the scalarized MDP for a given weight vector using vectorized VI."""
        weight = np.array(weight, dtype=np.float32)
        scalar_rewards = self.rewards @ weight
        V = np.zeros((self.rows, self.cols), dtype=np.float32)

        for _ in range(1000):
            next_V = V[self.transitions[..., 0], self.transitions[..., 1]]
            next_V = np.where(self.terminals, 0.0, next_V)
            Q = scalar_rewards + self.gamma * next_V
            new_V = np.max(Q, axis=-1)
            new_V[self.is_terminal_state] = 0.0

            delta = np.max(np.abs(V - new_V))
            V = new_V
            if delta < self.theta:
                break

        next_V = V[self.transitions[..., 0], self.transitions[..., 1]]
        next_V = np.where(self.terminals, 0.0, next_V)
        Q = scalar_rewards + self.gamma * next_V
        policy_array = np.argmax(Q + 1e-9 * self.rewards[..., 0], axis=-1)

        policy = {}
        for r, c in self.non_terminal_states:
            policy[(r, c)] = policy_array[r, c]
        return policy

    def reset_random(self, start_state=None):
        """
        Resets the environment according to the configured \rho distribution,
        or a specific state if provided.
        """
        self.env.reset()

        if start_state is None:
                                                                              
            idx = np.random.choice(self.n_states, p=self.rho)
            r, c = self.non_terminal_states[idx]
        else:
            r, c = start_state

                                                                       
        self.env.unwrapped.current_state = np.array([r, c], dtype=np.int32)

                                                           
        return np.array([r, c], dtype=np.int32), {}

    def get_average_return(self, policy):
        """Calculates the exact expected return J(\pi) under the distribution \rho."""
        R_pi = np.zeros((self.n_states, 2))
        P_pi = np.zeros((self.n_states, self.n_states))

        for i, (r, c) in enumerate(self.non_terminal_states):
            a = policy[(r, c)]
            R_pi[i] = self.rewards[r, c, a]
            if not self.terminals[r, c, a]:
                nr, nc = self.transitions[r, c, a]
                j = self.state_to_idx[(nr, nc)]
                P_pi[i, j] = 1.0

        A = np.eye(self.n_states) - self.gamma * P_pi
        V = np.linalg.solve(A, R_pi)

                                                                       
        ret = np.dot(self.rho, V)
        return (float(ret[0]), float(ret[1]))

    def _get_policy_matrices(self, policy):
        """Helper to extract the P and R matrices for Sherman-Morrison updates."""
        R_pi = np.zeros((self.n_states, 2))
        P_pi = np.zeros((self.n_states, self.n_states))

        for i, (r, c) in enumerate(self.non_terminal_states):
            a = policy[(r, c)]
            R_pi[i] = self.rewards[r, c, a]
            if not self.terminals[r, c, a]:
                nr, nc = self.transitions[r, c, a]
                j = self.state_to_idx[(nr, nc)]
                P_pi[i, j] = 1.0

        return P_pi, R_pi

    def find_exact_pareto_front(self):
        """
        Algorithm 1: The Exact Deterministic Pareto Front (10 points).
        Uses pure Pareto filtering to catch non-convex dents.
        Optimized with Sherman-Morrison O(S) rank-1 updates.
        """

        def hash_policy(p):
            return tuple(p[s] for s in self.non_terminal_states)

        initial_policy = self.solve([1.0, 0.0])
        initial_ret_t = self.get_average_return(initial_policy)
        initial_hash = hash_policy(initial_policy)

        found_returns = {initial_ret_t: [initial_policy]}
        queue = [initial_policy]
        processed_policies = {initial_hash}

        print("\n[Phase 1] Searching Exact Deterministic Pareto Front...")
        pbar = tqdm(total=1, desc="Deterministic Returns Found: 1")

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

                                        
            for idx, (r, c) in enumerate(self.non_terminal_states):
                state = (r, c)
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

                                                    
                    R_new_i = self.rewards[r, c, action]
                    P_new_i = np.zeros(self.n_states)
                    if not self.terminals[r, c, action]:
                        nr, nc = self.transitions[r, c, action]
                        P_new_i[self.state_to_idx[(nr, nc)]] = 1.0

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

                                                           
                    existing_rets = list(found_returns.keys())
                    nx, ny = n_ret_t

                    is_dominated = False
                    for ex, ey in existing_rets:
                        if ex >= nx and ey >= ny and (ex > nx or ey > ny):
                            is_dominated = True
                            break

                    if is_dominated:
                        continue

                                                       
                    to_remove = []
                    for ex, ey in existing_rets:
                        if nx >= ex and ny >= ey and (nx > ex or ny > ey):
                            to_remove.append((ex, ey))

                    for r_t in to_remove:
                        del found_returns[r_t]

                    if n_ret_t not in found_returns:
                        found_returns[n_ret_t] = [n_pol]
                        queue.append(n_pol)
                        pbar.set_description(
                            f"Deterministic Returns Found: {len(found_returns)}"
                        )
                    else:
                        found_returns[n_ret_t].append(n_pol)
                        queue.append(n_pol)

            pbar.total = len(queue)
            pbar.update(1)

        pbar.close()

        results = []
        for r_t, policies in found_returns.items():
            for policy in policies:
                results.append((np.array(r_t), policy))
        results.sort(key=lambda x: (x[0][0], x[0][1]))
        return results

    def find_ccs_ols(self):
        """
        Optimistic Linear Support (OLS) Algorithm.
        Guarantees the Exact Convex Pareto Front (CCS) without BFS disconnection issues.
        Runs Value Iteration strictly on the optimal faces.
        """
        print("Running Optimistic Linear Support (OLS) to find Exact CCS...")

                                           
        pol_top_left = self.solve([1.0, 0.0])
        ret_top_left = self.get_average_return(pol_top_left)

        pol_bottom_right = self.solve([0.0, 1.0])
        ret_bottom_right = self.get_average_return(pol_bottom_right)

                                                                         
        ccs_returns = {ret_top_left: pol_top_left, ret_bottom_right: pol_bottom_right}

                                                                              
        queue = [(ret_top_left, ret_bottom_right)]

        while queue:
            J1, J2 = queue.pop(0)

                                                                                  
                                           
                                                                    
            wx = J2[1] - J1[1]
            wy = J1[0] - J2[0]

                                         
            w_sum = wx + wy
            if w_sum <= 0:
                continue
            w = np.array([wx / w_sum, wy / w_sum])

                                                                  
            pol_new = self.solve(w)
            J_new = self.get_average_return(pol_new)

                                                                         
            current_edge_val = np.dot(w, J1)
            new_val = np.dot(w, J_new)

                                                                                        
            if new_val > current_edge_val + 1e-5:
                if J_new not in ccs_returns:
                    ccs_returns[J_new] = pol_new

                                                                           
                    queue.append((J1, J_new))
                    queue.append((J_new, J2))

                                                           
        results = [(np.array(r), p) for r, p in ccs_returns.items()]
        results.sort(key=lambda x: (x[0][0], x[0][1]))

        return results

    def filter_convex_coverage_set(self, pareto_results):
        """
        Algorithm 2: The Exact Convex Pareto Front (8 points).
        Takes the deterministic Pareto front and mathematically drops
        the concave dents to return the Exact Convex Coverage Set (CCS).
        """
        print("\n[Phase 2] Filtering for Convex Coverage Set (CCS)...")

                                
        unique_rets = list({tuple(r) for r, p in pareto_results})
        ccs_rets = set(unique_rets)

        for nx, ny in unique_rets:
            is_dented = False

                                                                       
            for i in range(len(unique_rets)):
                for j in range(i + 1, len(unique_rets)):
                    Ax, Ay = unique_rets[i]
                    Bx, By = unique_rets[j]

                    if Ax > Bx:
                        Ax, Ay, Bx, By = Bx, By, Ax, Ay

                                                                            
                    if Ax < nx < Bx:
                                                                                
                        t = (nx - Ax) / (Bx - Ax)
                        y_line = Ay + t * (By - Ay)

                                                                              
                        if ny < y_line - 1e-6:
                            is_dented = True
                            break
                if is_dented:
                    break

            if is_dented:
                ccs_rets.remove((nx, ny))

                                                                  
        ccs_results = [res for res in pareto_results if tuple(res[0]) in ccs_rets]
        return ccs_results

    def find_fast_ccs_front(self):
        """
        Exhaustively finds the Exact Convex Pareto Front (8 points).
        Filters strictly INSIDE the loop.
        Eliminates the BFS queue explosion by not branching from redundant aliases.
        """
        from tqdm import tqdm

        def hash_policy(p):
            return tuple(p[s] for s in self.non_terminal_states)

        def is_ccs_dominated_2d(new_p, existing_rets, tolerance=1e-6):
            nx, ny = new_p
                                  
            for ex, ey in existing_rets:
                if ex >= nx and ey >= ny and (ex > nx or ey > ny):
                    return True
                                          
            for i in range(len(existing_rets)):
                for j in range(i + 1, len(existing_rets)):
                    Ax, Ay = existing_rets[i]
                    Bx, By = existing_rets[j]
                    if Ax > Bx:
                        Ax, Ay, Bx, By = Bx, By, Ax, Ay
                    if Ax < nx < Bx:
                        t = (nx - Ax) / (Bx - Ax)
                        y_line = Ay + t * (By - Ay)
                        if ny <= y_line + tolerance:
                            return True
            return False

                       
        initial_policy = self.solve([1.0, 0.0])
        initial_ret_t = self.get_average_return(initial_policy)
        initial_hash = hash_policy(initial_policy)

        found_returns = {initial_ret_t: [initial_policy]}
        queue = [initial_policy]
        processed_policies = {initial_hash}

        print("Running Ultra-Fast In-Loop CCS Search...")
        pbar = tqdm(total=1, desc="CCS Vertices found: 1")

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

                                    
            for idx, (r, c) in enumerate(self.non_terminal_states):
                state = (r, c)
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

                                                    
                    R_new_i = self.rewards[r, c, action]
                    P_new_i = np.zeros(self.n_states)
                    if not self.terminals[r, c, action]:
                        nr, nc = self.transitions[r, c, action]
                        P_new_i[self.state_to_idx[(nr, nc)]] = 1.0

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

                                                      
                    existing_rets = list(found_returns.keys())
                    nx, ny = n_ret_t

                    if is_ccs_dominated_2d(n_ret_t, existing_rets):
                        continue

                                                    
                    to_remove = set()
                    for ex, ey in existing_rets:
                        if nx >= ex and ny >= ey and (nx > ex or ny > ey):
                            to_remove.add((ex, ey))
                            continue
                                                                            
                        for ox, oy in existing_rets:
                            if (ox, oy) == (ex, ey):
                                continue
                            Ax, Ay, Bx, By = nx, ny, ox, oy
                            if Ax > Bx:
                                Ax, Ay, Bx, By = Bx, By, Ax, Ay
                            if Ax < ex < Bx:
                                t = (ex - Ax) / (Bx - Ax)
                                y_line = Ay + t * (By - Ay)
                                if ey <= y_line + 1e-6:
                                    to_remove.add((ex, ey))
                                    break

                    for r_t in to_remove:
                        del found_returns[r_t]

                                                                      
                    if n_ret_t not in found_returns:
                                                          
                        found_returns[n_ret_t] = [n_pol]
                        queue.append(n_pol)
                        pbar.set_description(
                            f"Deterministic Returns Found: {len(found_returns)}"
                        )
                    else:
                                                                                  
                        found_returns[n_ret_t].append(n_pol)
                        queue.append(n_pol)

            pbar.total = len(queue)
            pbar.update(1)

        pbar.close()

        results = []
        for r_t, policies in found_returns.items():
            for policy in policies:
                results.append((np.array(r_t), policy))
        results.sort(key=lambda x: (x[0][0], x[0][1]))
        return results

    def run_full_analysis(self):
        """Wrapper to run the full pipeline and print the comparative results."""
                                         
        deterministic_front = self.find_exact_pareto_front()
        unique_det_returns = {tuple(r) for r, p in deterministic_front}

                                                       
        convex_front = self.filter_convex_coverage_set(deterministic_front)
        unique_ccs_returns = {tuple(r) for r, p in convex_front}

        print("\n=== FINAL ANALYSIS ===")
        print(f"Total Aliased Policies Found: {len(deterministic_front)}")
        print(
            f"Exact Deterministic Returns (The full curve): {len(unique_det_returns)} points"
        )
        print(
            f"Exact Convex Returns (The bridged shell): {len(unique_ccs_returns)} points"
        )

        dented_points = unique_det_returns - unique_ccs_returns
        if dented_points:
            print(
                f"\nDiscovered {len(dented_points)} concave 'dented' returns that Linear Scalarization misses:"
            )
            for pt in sorted(list(dented_points)):
                print(f"  -> {pt}")

        return deterministic_front, convex_front
