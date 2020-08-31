import multiprocessing
import matplotlib.pyplot as plt
import numpy as np


class RunningAverage:

    def __init__(self):
        self.n = 0
        self.avg = 0

    def update(self, value):
        self.n += 1
        self.avg += (1 / self.n) * (value - self.avg)


class MultiArmBandit:

    def __init__(self, k_arms, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma
        self.random_state = np.random.RandomState()
        self.q_true = self.random_state.normal(self.mu, self.sigma, k_arms)

    def get_reward_for_arm(self, k_arm):
        mu = self.q_true[k_arm]
        sigma = 1.0
        return self.random_state.normal(mu, sigma)


def run_one_bandit(k_arms, n_steps, epsilon):
    bandit = MultiArmBandit(k_arms)
    random_state = np.random.RandomState()
    q_estimates = np.zeros(k_arms)
    a_counts = np.zeros(k_arms)
    rewards = np.zeros(n_steps)
    for i_step in range(n_steps):

        # choose action
        #--------------------------------------------------------
        if random_state.uniform() > epsilon:
            # exploit = choose argmax (break ties randomly)
#            q_max = np.max(q_estimates)
#            i_action = random_state.choice([
#                ii for ii, qq in enumerate(q_estimates) if qq == q_max])
            i_action = random_state.choice(
                np.argwhere(q_estimates == np.max(q_estimates)).squeeze(axis=1)
            )
        else:
            # explore = choose random
            i_action = random_state.randint(low=0, high=k_arms)

        # get reward and update estimates
        #--------------------------------------------------------
        reward = bandit.get_reward_for_arm(i_action)
        a_counts[i_action] += 1

        step = 1 / a_counts[i_action]
        q_estimates[i_action] += step * (reward - q_estimates[i_action])
        rewards[i_step] = reward

    return {
        "bandit": bandit.q_true,
        "rewards": rewards,
    }



def get_results_for_n_trials_mp(n_trials, k_arms, n_steps, epsilon, n_cores):

    print(f"performing {n_trials} trials with {n_cores} cores.")
    experiment_args = [
        (k_arms, n_steps, epsilon) for ii in range(n_trials)]
    p = multiprocessing.Pool(n_cores)
    results = p.starmap(run_one_bandit, experiment_args)
    return results


def run_experiment_mp(n_trials, k_arms, n_steps, epsilons, n_cores):
    experiment = {}
    for epsilon in epsilons:
        results = get_results_for_n_trials_mp(n_trials, k_arms, n_steps, epsilon, n_cores)
        avg_rwd_at_step = np.zeros(n_steps)
        for result in results:
            avg_rwd_at_step += result['rewards']
        avg_rwd_at_step /= n_trials
        experiment[epsilon] = avg_rwd_at_step
    return experiment


def run_experiment_sp(n_trials, k_arms, n_steps, epsilons):
    experiment = {}
    for epsilon in epsilons:
        print(f"performing {n_trials} trials with 1 core.")
        avg_rwd_at_step = np.zeros(n_steps)
        for ii in range(n_trials):
            result = run_one_bandit(k_arms, n_steps, epsilon)
            avg_rwd_at_step += result['rewards']
        avg_rwd_at_step /= n_trials
        experiment[epsilon] = avg_rwd_at_step
    return experiment




if __name__ == "__main__":

    np.random.seed(2937)
    n_trials = 2000
    k_arms = 10
    n_steps = 1000
    n_cores = multiprocessing.cpu_count()

    epsilons = [0.0, 0.01, 0.1]
    colors = ["green", "red", "blue"]

    experiment_mp = run_experiment_mp(n_trials, k_arms, n_steps, epsilons, n_cores)
    fig = plt.figure()
    for eps, clr in zip(epsilons, colors):
        plt.plot(experiment_mp[eps], color=clr)

#    experiment_sp = run_experiment_sp(n_trials, k_arms, n_steps, epsilons)
#    fig = plt.figure()
#    for eps, clr in zip(epsilons, colors):
#        plt.plot(experiment_sp[eps], color=clr)
