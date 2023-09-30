from models.RNN import RNN
from oort_client import Client
from oort_server import Server
from oort_trainer import Trainer
from WESAD import WESAD

def main():
    model=  RNN#chosen
    datasource = WESAD
    trainer = Trainer
    client = Client(model=model, datasource=datasource, trainer=trainer)
    server = Server(model=model, datasource=datasource, trainer=trainer)
    server.run(client)

if __name__ == "__main__":
    main()
