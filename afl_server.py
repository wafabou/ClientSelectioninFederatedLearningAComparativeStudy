import logging
import math
import random

import numpy as np
from plato.config import Config
from plato.servers import fedavg


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

        self.local_values = {}

    def weights_aggregated(self, updates):
        for update in updates:
            self.local_values[update.client_id]["valuation"] = update.report.valuation

    def calc_sample_distribution(self, clients_pool):
        for client_id in clients_pool:
            if client_id not in self.local_values:
                self.local_values[client_id] = {}
                self.local_values[client_id]["valuation"] = -float("inf")
                self.local_values[client_id]["prob"] = 0.0
        num_smallest = int(Config().algorithm.alpha1 * len(clients_pool))
        smallest_valuations = dict(
            sorted(self.local_values.items(), key=lambda item: item[1]["valuation"])[
                :num_smallest
            ]
        )

        for client_id in smallest_valuations.keys():
            self.local_values[client_id]["valuation"] = -float("inf")

        for client_id in clients_pool:
            self.local_values[client_id]["prob"] = math.exp(
                Config().algorithm.alpha2 * self.local_values[client_id]["valuation"]
            )

    def choose_clients(self, clients_pool, clients_count):
        assert clients_count <= len(clients_pool)
        random.setstate(self.prng_state)
        self.calc_sample_distribution(clients_pool)
        num1 = int(math.floor((1 - Config().algorithm.alpha3) * clients_count))
        probs = np.array(
            [self.local_values[client_id]["prob"] for client_id in clients_pool]
        )
        probs = probs + 0.01
        probs /= probs.sum()

        subset1 = np.random.choice(clients_pool, num1, p=probs, replace=False).tolist()
        num2 = clients_count - num1
        remaining = clients_pool.copy()
        for client_id in subset1:
            remaining.remove(client_id)
        subset2 = random.sample(remaining, num2)
        selected_clients = subset1 + subset2
        self.prng_state = random.getstate()
        logging.info("[%s] Selected clients: %s", self, selected_clients)
        return selected_clients
