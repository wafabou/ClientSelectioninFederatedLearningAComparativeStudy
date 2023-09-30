from types import SimpleNamespace

import numpy as np
from plato.clients import simple


class Client(simple.Client):
    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:

        train_squared_loss_step = self.trainer.run_history.get_metric_values(
            "train_squared_loss_step"
        )
        report.statistical_utility = report.num_samples * np.sqrt(
            1.0 / report.num_samples * sum(train_squared_loss_step)
        )

        return report
