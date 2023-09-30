
import logging
import math
import random
import numpy as np

from plato.servers import fedavg
from plato.config import Config


class Server(fedavg.Server):

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )
        self.blacklist = []
        self.client_utilities = {}
        self.client_durations = {}
        self.client_last_rounds = {}
        self.client_selected_times = {}
        self.desired_duration = Config().server.desired_duration
        self.explored_clients = []
        self.unexplored_clients = []
        self.exploration_factor = Config().server.exploration_factor
        self.step_window = Config().server.step_window
        self.pacer_step = Config().server.desired_duration
        self.penalty = Config().server.penalty
        self.util_history = []
        self.cut_off = (
            Config().server.cut_off if hasattr(Config().server, "cut_off") else 0.95
        )
        self.blacklist_num = (
            Config().server.blacklist_num
            if hasattr(Config().server, "blacklist_num")
            else 10
        )

    def configure(self) -> None:
        """Initialize necessary variables."""
        super().configure()

        self.client_utilities = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }
        self.client_durations = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }
        self.client_last_rounds = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }
        self.client_selected_times = {
            client_id: 0 for client_id in range(1, self.total_clients + 1)
        }

        self.unexplored_clients = list(range(1, self.total_clients + 1))

    def weights_aggregated(self, updates):
        for update in updates:
            self.client_utilities[update.client_id] = update.report.statistical_utility
            self.client_durations[update.client_id] = update.report.training_time
            self.client_last_rounds[update.client_id] = self.current_round
            self.client_utilities[update.client_id] = self.calc_client_util(
                update.client_id
            )
        self.util_history.append(
            sum(update.report.statistical_utility for update in updates)
        )

        if self.current_round >= 2 * self.step_window:
            last_pacer_rounds = sum(
                self.util_history[-2 * self.step_window : -self.step_window]
            )
            current_pacer_rounds = sum(self.util_history[-self.step_window :])
            if last_pacer_rounds > current_pacer_rounds:
                self.desired_duration += self.pacer_step
        for update in updates:
            if self.client_selected_times[update.client_id] > self.blacklist_num:
                self.blacklist.append(update.client_id)

    def choose_clients(self, clients_pool, clients_count):
        """Choose a subset of the clients to participate in each round."""
        selected_clients = []

        if self.current_round > 1:
            exploited_clients_count = max(
                math.ceil((1.0 - self.exploration_factor) * clients_count),
                clients_count - len(self.unexplored_clients),
            )

            sorted_by_utility = sorted(
                self.client_utilities, key=self.client_utilities.get, reverse=True
            )
            sorted_by_utility = [
                client for client in sorted_by_utility if client in clients_pool
            ]

            cut_off_util = (
                self.client_utilities[sorted_by_utility[exploited_clients_count - 1]]
                * self.cut_off
            )
            exploited_clients = []
            for client_id in sorted_by_utility:
                if (
                    self.client_utilities[client_id] > cut_off_util
                    and client_id not in self.blacklist
                ):
                    exploited_clients.append(client_id)

            # Sample clients with their utilities
            total_utility = float(
                sum(self.client_utilities[client_id] for client_id in exploited_clients)
            )

            probabilities = [
                self.client_utilities[client_id] / total_utility
                for client_id in exploited_clients
            ]

            if len(probabilities) > 0 and exploited_clients_count > 0:
                selected_clients = np.random.choice(
                    exploited_clients,
                    min(len(exploited_clients), exploited_clients_count),
                    p=probabilities,
                    replace=False,
                )
                selected_clients = selected_clients.tolist()

            last_index = (
                sorted_by_utility.index(exploited_clients[-1])
                if exploited_clients
                else 0
            )
            if len(selected_clients) < exploited_clients_count:
                for index in range(last_index + 1, len(sorted_by_utility)):
                    if (
                        not sorted_by_utility[index] in self.blacklist
                        and len(selected_clients) < exploited_clients_count
                    ):
                        selected_clients.append(sorted_by_utility[index])

        random.setstate(self.prng_state)

        selected_unexplore_clients = random.sample(
            self.unexplored_clients, clients_count - len(selected_clients)
        )

        self.prng_state = random.getstate()
        self.explored_clients += selected_unexplore_clients

        for client_id in selected_unexplore_clients:
            self.unexplored_clients.remove(client_id)

        selected_clients += selected_unexplore_clients

        for client in selected_clients:
            self.client_selected_times[client] += 1

        logging.info("[%s] Selected clients: %s", self, selected_clients)

        return selected_clients

    def calc_client_util(self, client_id):
        client_utility = self.client_utilities[client_id] + math.sqrt(
            0.1 * math.log(self.current_round) / self.client_last_rounds[client_id]
        )

        if self.desired_duration < self.client_durations[client_id]:
            global_utility = (
                self.desired_duration / self.client_durations[client_id]
            ) ** self.penalty
            client_utility *= global_utility

        return client_utility
