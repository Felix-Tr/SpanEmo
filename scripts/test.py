"""
Usage:
    main.py [options]

Options:
    -h --help                         show this screen
    --model-path=<str>                path of the trained model
    --max-length=<int>                text length [default: 128]
    --seed=<int>                      seed [default: 0]
    --test-batch-size=<int>           batch size [default: 32]
    --lang=<str>                      language choice [default: English]
    --test-path=<str>                 file path of the test set [default: ]
    --wandb=<str>                     wandb configuration for debugging [default: '']
    --run=<str>                       wandb tag for run [default: '']
"""
from learner import EvaluateOnTest
from model import SpanEmo
from data_loader import DataClass
from torch.utils.data import DataLoader
import torch
from docopt import docopt
import numpy as np

from time import localtime
import wandb

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
# Define Dataloaders
#####################################################################
test_dataset = DataClass(args, args['--test-path'])
test_data_loader = DataLoader(test_dataset,
                              batch_size=int(args['--test-batch-size']),
                              shuffle=False)
print('The number of Test batches: ', len(test_data_loader))
#############################################################################
# Run the model on a Test set
#############################################################################
model = SpanEmo(lang=args['--lang'])
learn = EvaluateOnTest(model, test_data_loader, model_path=args['--model-path'])
learn.predict(device=device)


