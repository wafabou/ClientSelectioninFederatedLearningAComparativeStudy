import warnings
import asyncio
from models.RNN import RNN
from pisces_client import Client
from pisces_server import Server
from pisces_trainer import Trainer
from WESAD import WESAD

warnings.filterwarnings("ignore", category=UserWarning)
def main():
    model=  RNN
    datasource = WESAD
    trainer = Trainer
    client = Client(model=model, datasource=datasource, trainer=trainer)
    server = Server(model=model, datasource=datasource, trainer=trainer)
    server.run(client)

if __name__ == "__main__":
    main()
