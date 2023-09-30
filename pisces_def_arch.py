from functools import partial

from torch import nn
from torch.nn.modules.activation import LeakyReLU
from WESAD import WESAD
from pisces_server import Server
from pisces_trainer import Trainer
from pisces_client import Client

from keras.models import Sequential

from keras.layers import Dense, Embedding, Flatten
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from keras.layers import LSTM


num_neurons = 30
num_features = 30
num_classes = 10

modelstress = partial(
                nn.Sequential,
                nn.Linear(46,128),
                #nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(128, 256),
                #nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 3 ),
                #nn.Dropout(0.5),
                nn.LogSoftmax(dim=1),
    )


def main():
    datasource = WESAD
    trainer = Trainer
    client = Client(model=modelstress, datasource=datasource, trainer=trainer)
    server = Server(model=modelstress, datasource=datasource, trainer=trainer)
    server.run(client)

if __name__ == "__main__":
    main()
