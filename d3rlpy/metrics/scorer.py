from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, cast

import gym
import numpy as np
from typing_extensions import Protocol

from ..dataset import Episode, TransitionMiniBatch
from ..preprocessing.reward_scalers import RewardScaler
from ..preprocessing.stack import StackedObservation

WINDOW_SIZE = 1024


class AlgoProtocol(Protocol):
    def predict(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        ...

    def predict_value(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        with_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        ...

    @property
    def n_frames(self) -> int:
        ...

    @property
    def gamma(self) -> float:
        ...

    @property
    def reward_scaler(self) -> Optional[RewardScaler]:
        ...


class DynamicsProtocol(Protocol):
    def predict(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        with_variance: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        ...

    @property
    def n_frames(self) -> int:
        ...

    @property
    def reward_scaler(self) -> Optional[RewardScaler]:
        ...


def _make_batches(
    episode: Episode, window_size: int, n_frames: int
) -> Iterator[TransitionMiniBatch]:
    n_batches = len(episode) // window_size
    if len(episode) % window_size != 0:
        n_batches += 1
    for i in range(n_batches):
        head_index = i * window_size
        last_index = min(head_index + window_size, len(episode))
        transitions = episode.transitions[head_index:last_index]
        batch = TransitionMiniBatch(transitions, n_frames)
        yield batch

# Attempt: Calculate the true total discounted reward by taking initial action a in initial state s.
def true_q_value_scorer(algo: AlgoProtocol, episodes: List[Episode]) -> float:
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            # estimate values for next observations
            next_actions = algo.predict([batch.next_observations[0]])
            next_values = algo.predict_value(
                [batch.next_observations[0]], next_actions
            )
            mask = (1.0 - np.asarray(batch.terminals)).reshape(-1)
            rewards = np.asarray(batch.next_rewards).reshape(-1)
            if algo.reward_scaler:
                rewards = algo.reward_scaler.transform_numpy(rewards)
            y = rewards + algo.gamma * cast(np.ndarray, next_values) * mask
    return float(np.mean(y))


def td_error_scorer(algo: AlgoProtocol, episodes: List[Episode]) -> float:
    r"""Returns average TD error.
    Args:
        algo: algorithm.
        episodes: list of episodes.
    Returns:
        average TD error.
    """
    total_errors = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            # estimate values for current observations
            values = algo.predict_value(batch.observations, batch.actions)

            # estimate values for next observations
            next_actions = algo.predict(batch.next_observations)
            next_values = algo.predict_value(
                batch.next_observations, next_actions
            )

            # calculate td errors
            mask = (1.0 - np.asarray(batch.terminals)).reshape(-1)
            rewards = np.asarray(batch.next_rewards).reshape(-1)
            if algo.reward_scaler:
                rewards = algo.reward_scaler.transform_numpy(rewards)
            y = rewards + algo.gamma * cast(np.ndarray, next_values) * mask
            total_errors += ((values - y) ** 2).tolist()

    return float(np.mean(total_errors))


def discounted_sum_of_advantage_scorer(
    algo: AlgoProtocol, episodes: List[Episode]
) -> float:
    r"""Returns average of discounted sum of advantage
    """
    total_sums = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            # estimate values for dataset actions
            dataset_values = algo.predict_value(
                batch.observations, batch.actions
            )
            dataset_values = cast(np.ndarray, dataset_values)

            # estimate values for the current policy
            actions = algo.predict(batch.observations)
            on_policy_values = algo.predict_value(batch.observations, actions)

            # calculate advantages
            advantages = (dataset_values - on_policy_values).tolist()

            # calculate discounted sum of advantages
            A = advantages[-1]
            sum_advantages = [A]
            for advantage in reversed(advantages[:-1]):
                A = advantage + algo.gamma * A
                sum_advantages.append(A)

            total_sums += sum_advantages

    # smaller is better
    return float(np.mean(total_sums))


def average_value_estimation_scorer(
    algo: AlgoProtocol, episodes: List[Episode]
) -> float:
    r"""Returns average value estimation
    """
    total_values = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            actions = algo.predict(batch.observations)
            values = algo.predict_value(batch.observations, actions)
            total_values += cast(np.ndarray, values).tolist()
    return float(np.mean(total_values))


def value_estimation_std_scorer(
    algo: AlgoProtocol, episodes: List[Episode]
) -> float:
    r"""Returns standard deviation of value estimation.
    """
    total_stds = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            actions = algo.predict(batch.observations)
            _, stds = algo.predict_value(batch.observations, actions, True)
            total_stds += stds.tolist()
    return float(np.mean(total_stds))


def initial_state_value_estimation_scorer(
    algo: AlgoProtocol, episodes: List[Episode]
) -> float:
    r"""Returns mean estimated action-values at the initial states
    """
    total_values = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            # estimate action-value in initial states
            actions = algo.predict([batch.observations[0]])
            values = algo.predict_value([batch.observations[0]], actions)
            total_values.append(values[0])
    return float(np.mean(total_values))


def soft_opc_scorer(
    return_threshold: float,
) -> Callable[[AlgoProtocol, List[Episode]], float]:
    r"""Returns Soft Off-Policy Classification metrics
    """

    def scorer(algo: AlgoProtocol, episodes: List[Episode]) -> float:
        success_values = []
        all_values = []
        for episode in episodes:
            is_success = episode.compute_return() >= return_threshold
            for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
                values = algo.predict_value(batch.observations, batch.actions)
                values = cast(np.ndarray, values)
                all_values += values.reshape(-1).tolist()
                if is_success:
                    success_values += values.reshape(-1).tolist()
        return float(np.mean(success_values) - np.mean(all_values))

    return scorer


def continuous_action_diff_scorer(
    algo: AlgoProtocol, episodes: List[Episode]
) -> float:
    r"""Returns squared difference of actions between algorithm and dataset
    """
    total_diffs = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            actions = algo.predict(batch.observations)
            diff = ((batch.actions - actions) ** 2).sum(axis=1).tolist()
            total_diffs += diff
    return float(np.mean(total_diffs))


def discrete_action_match_scorer(
    algo: AlgoProtocol, episodes: List[Episode]
) -> float:
    r"""Returns percentage of identical actions between algorithm and dataset
    """
    total_matches = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            actions = algo.predict(batch.observations)
            match = (batch.actions.reshape(-1) == actions).tolist()
            total_matches += match
    return float(np.mean(total_matches))


def evaluate_on_environment(
    env: gym.Env, n_trials: int = 10, epsilon: float = 0.0, render: bool = False
) -> Callable[..., float]:
    """Returns scorer function of evaluation on environment
    """

    # for image observation
    observation_shape = env.observation_space.shape
    is_image = len(observation_shape) == 3

    def scorer(algo: AlgoProtocol, *args: Any) -> float:
        if is_image:
            stacked_observation = StackedObservation(
                observation_shape, algo.n_frames
            )

        episode_rewards = []
        for _ in range(n_trials):
            observation = env.reset()
            episode_reward = 0.0

            # frame stacking
            if is_image:
                stacked_observation.clear()
                stacked_observation.append(observation)

            while True:
                # take action
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    if is_image:
                        action = algo.predict([stacked_observation.eval()])[0]
                    else:
                        action = algo.predict([observation])[0]

                observation, reward, done, _ = env.step(action)
                episode_reward += reward

                if is_image:
                    stacked_observation.append(observation)

                if render:
                    env.render()

                if done:
                    break
            episode_rewards.append(episode_reward)
        return float(np.mean(episode_rewards))

    return scorer


def dynamics_observation_prediction_error_scorer(
    dynamics: DynamicsProtocol, episodes: List[Episode]
) -> float:
    r"""Returns MSE of observation prediction
    """
    total_errors = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, dynamics.n_frames):
            pred = dynamics.predict(batch.observations, batch.actions)
            errors = ((batch.next_observations - pred[0]) ** 2).sum(axis=1)
            total_errors += errors.tolist()
    return float(np.mean(total_errors))


def dynamics_reward_prediction_error_scorer(
    dynamics: DynamicsProtocol, episodes: List[Episode]
) -> float:
    r"""Returns MSE of reward prediction
    """
    total_errors = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, dynamics.n_frames):
            pred = dynamics.predict(batch.observations, batch.actions)
            rewards = batch.next_rewards
            if dynamics.reward_scaler:
                rewards = dynamics.reward_scaler.transform_numpy(rewards)
            errors = ((rewards - pred[1]) ** 2).reshape(-1)
            total_errors += errors.tolist()
    return float(np.mean(total_errors))


def dynamics_prediction_variance_scorer(
    dynamics: DynamicsProtocol, episodes: List[Episode]
) -> float:
    """Returns prediction variance of ensemble dynamics
    """
    total_variances = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, dynamics.n_frames):
            pred = dynamics.predict(batch.observations, batch.actions, True)
            pred = cast(Tuple[np.ndarray, np.ndarray, np.ndarray], pred)
            total_variances += pred[2].tolist()
    return float(np.mean(total_variances))
