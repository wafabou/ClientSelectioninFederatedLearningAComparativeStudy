import afl_client
import afl_server
import asyncio
from afl_Trainer import  Trainer
from models.RNN import RNN
from WESAD import WESAD

def main():
    model=  RNN
    datasource = WESAD
    trainer = Trainer
    client = afl_client.Client(model=model, datasource=datasource)
    server = afl_server.Server(model=model, datasource=datasource)
    server.run(client)
if __name__ == "__main__":
    main()
