"""
Usage:
    main.py [options]

Options:
    -h --help                         show this screen
    --loss-type=<str>                 Which loss to use cross-ent|corr|joint. [default: cross-entropy]
    --max-length=<int>                text length [default: 128]
    --output-dropout=<float>          prob of dropout applied to the output layer [default: 0.1]
    --seed=<int>                      fixed random seed number [default: 42]
    --train-batch-size=<int>          batch size [default: 32]
    --eval-batch-size=<int>           batch size [default: 32]
    --max-epoch=<int>                 max epoch [default: 20]
    --ffn-lr=<float>                  ffn learning rate [default: 0.001]
    --bert-lr=<float>                 bert learning rate [default: 2e-5]
    --lang=<str>                      language choice [default: English]
    --dev-path=<str>                  file path of the dev set [default: '']
    --train-path=<str>                file path of the train set [default: '']
    --alpha-loss=<float>              weight used to balance the loss [default: 0.2]
    --wandb=<str>                     wandb configuration for debugging [default: '']
    --model-path=<str>                path to save checkpoints to [default: '']
    --from-pretrained=<str>           path to state file to load from pretrained [default: '']
    --run=<str>                       wandb tag for run [default: '']
"""

from learner import Trainer
from model import SpanEmo
from data_loader import DataClass
from torch.utils.data import DataLoader
import torch
from docopt import docopt
import datetime
import json
import numpy as np
import wandb
import os

from time import localtime

args = docopt(__doc__)

cur_time = f"{localtime().tm_mday}.{localtime().tm_mon} - {localtime().tm_hour}:{localtime().tm_min}"
run = args["--run"] + " train - " + cur_time

# wandb integration
if args["--wandb"] == "debug":
    os.environ['WANDB_DISABLED'] = 'true'
wandb.login(key="a6da9e40226ee6796df369e63bf8ee32a1171278")
wandb.init(project=f"SpanEmo-bert", name=run)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if str(device) == 'cuda:0':
    print("Currently using GPU: {}".format(device))
    np.random.seed(int(args['--seed']))
    torch.cuda.manual_seed_all(int(args['--seed']))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    print("Currently using CPU")
#####################################################################
# Save hyper-parameter values ---> config.json
# Save model weights ---> filename.pt using current time
#####################################################################
now = datetime.datetime.now()
filename = now.strftime("%Y-%m-%d-%H:%M:%S") + args["--run"]
fw = open('configs/' + filename + '.json', 'a')
model_path = filename + '.pt'
args['--checkpoint-path'] = model_path
json.dump(args, fw, sort_keys=True, indent=2)
#####################################################################
# Define Dataloaders
#####################################################################
train_dataset = DataClass(args, args['--train-path'])
train_data_loader = DataLoader(train_dataset,
                               batch_size=int(args['--train-batch-size']),
                               shuffle=True
                               )
print('The number of training batches: ', len(train_data_loader))
dev_dataset = DataClass(args, args['--dev-path'])
dev_data_loader = DataLoader(dev_dataset,
                             batch_size=int(args['--eval-batch-size']),
                             shuffle=False
                             )
print('The number of validation batches: ', len(dev_data_loader))
#############################################################################
# Define Model & Training Pipeline
#############################################################################
model = SpanEmo(output_dropout=float(args['--output-dropout']),
                lang=args['--lang'],
                joint_loss=args['--loss-type'],
                alpha=float(args['--alpha-loss']))
if args["--from-pretrained"] != "''":
    model.to(device).load_state_dict(torch.load(args["--from-pretrained"]))
#############################################################################
# Start Training
#############################################################################
learn = Trainer(model, train_data_loader, dev_data_loader, filename, args)
learn.fit(
    num_epochs=int(args['--max-epoch']),
    args=args,
    device=device
)
