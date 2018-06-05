import json

import click
import matplotlib

matplotlib.use('Agg')
from cfcm.zoo.fitrunners import (
    LSTMResNetMontgomeryXray,
    LSTMResNetEndovis,
    ResNetEndovis,
    ResNetMontgomeryXray
)


@click.group()
def cli():
    pass


@click.command()
@click.argument('json_file', type=click.Path(exists=True))
@click.option('--data_folder', default='/data')
@click.option('--model_folder', default='/data/model')
def train(json_file, data_folder, model_folder):
    if json_file is None:
        raise RuntimeError('missing json file')

    jdata = json.load(open(json_file))

    model_name = jdata['algorithm']['name']
    data_name = jdata['dataset']['name']
    number_epochs = jdata['algorithm']['n_epochs']

    if model_name == 'LSTMResNet':
        if data_name == 'ENDOVIS':
            fit_runner = LSTMResNetEndovis(jdata, data_folder=data_folder, model_folder=model_folder)
        elif data_name == 'Montgomery':
            fit_runner = LSTMResNetMontgomeryXray(jdata, data_folder=data_folder, model_folder=model_folder)

    if model_name == 'ResNet':
        if data_name == 'ENDOVIS':
            fit_runner = ResNetEndovis(jdata, data_folder=data_folder, model_folder=model_folder)
        elif data_name == 'Montgomery':
            fit_runner = ResNetMontgomeryXray(jdata, data_folder=data_folder, model_folder=model_folder)

    for i in range(number_epochs):
        print('epoch {}'.format(i))
        fit_runner.run_epoch_train()
        fit_runner.run_epoch_valid()


cli.add_command(train)

if __name__ == '__main__':
    cli()
