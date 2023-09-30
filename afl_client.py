import logging
import math
from types import SimpleNamespace
from plato.clients import simple
from plato.utils import fonts


class Client(simple.Client):

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        loss = self.trainer.run_history.get_latest_metric("train_loss")
        logging.info(fonts.colourize(f"[Client #{self.client_id}] Loss value: {loss}"))
        report.valuation = self.calc_valuation(report.num_samples, loss)
        return report

    def calc_valuation(self, num_samples, loss):
        valuation = float(1 / math.sqrt(num_samples)) * loss
        return valuation
