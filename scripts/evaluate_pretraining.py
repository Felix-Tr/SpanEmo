from learner import Trainer, EvaluateOnTest
from model import SpanEmo
from data_loader import DataClass
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import datetime
import json
import numpy as np
import wandb
import os

args = {"--loss-type": "cross-entropy",
        "--max-length": 128,
        "--output-dropout": 0.1,
        "--seed": 42,
        "--train-batch-size": 8,
        "--eval-batch-size": 8,
        "--max-epoch": 3,
        "--ffn-lr": 0.001,
        "--bert-lr": 2e-5,
        "--lang": "English",
        "--dev-path": "",
        "--train-path": "",
        "--alpha-loss": 0.2,
        "--wandb": "debug",
        "--model-path": "",
        "--from-pretrained": "",
        "--run": ""}
"""
Test
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


from time import localtime

cur_time = f"{localtime().tm_mday}.{localtime().tm_mon} - {localtime().tm_hour}:{localtime().tm_min}"
run = args["--run"] + " train - " + cur_time

# wandb integration
if args["--wandb"] == "debug":
    os.environ['WANDB_DISABLED'] = 'true'
wandb.login(key="a6da9e40226ee6796df369e63bf8ee32a1171278")
project = f"SpanEmo-transfer"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if str(device) == 'cuda:0':
    print("Currently using GPU: {}".format(device))
    np.random.seed(int(args['--seed']))
    torch.cuda.manual_seed_all(int(args['--seed']))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    print("Currently using CPU")

from pathlib import Path
drive_ws = Path("/content/drive/MyDrive/temporary/masterthesis_drive")
runs_ws = drive_ws / "models" / "20220607" / "SpaEmo"
runs_ws.mkdir(parents=True, exist_ok=True)

langs = ["GermanSentiment"] #["English", "German", "GermanSentiment"]
cases = ["pretrained", "base"] #["pretrained", "base"]
fracs = [100, 200, 500, 1000, 5000] #[100, 200, 500, 1000, 5000]

for lang in langs:
    tag = "" if lang == "English" else "-DE"
    args["--lang"] = lang
    args["--dev-path"] = drive_ws / "SpanEmoData" / "E-c" / f"2018-E-c-En-dev{tag}.txt"
    args["--train-path"] = drive_ws / "SpanEmoData" / "E-c" / f"2018-E-c-En-train{tag}.txt"

    #####################################################################
    # Define Dataloaders
    #####################################################################
    train_dataset = DataClass(args, args['--train-path'])
    # train_data_loader = DataLoader(train_dataset,
    #                                batch_size=int(args['--train-batch-size']),
    #                                shuffle=True
    #                                )
    # print('The number of training batches: ', len(train_data_loader))
    dev_dataset = DataClass(args, args['--dev-path'])
    dev_data_loader = DataLoader(dev_dataset,
                                 batch_size=int(args['--eval-batch-size']),
                                 shuffle=False
                                 )
    print('The number of validation batches: ', len(dev_data_loader))

    print(f"################################### Running: {lang} #######################################")
    for case in cases:
        print(f"################################### Evaluating: {case} #######################################")
        for frac in tqdm(fracs, desc="Fractions: "):

            name = f"SpanEmo-{lang}-{case}-{frac}"

            model_dir = runs_ws / name
            model_dir.mkdir(parents=True, exist_ok=True)
            args["--model-path"] = str(model_dir)
            train_dataset_frac = torch.utils.data.Subset(train_dataset, list(range(frac)))
            train_data_loader = DataLoader(train_dataset_frac,
                                           batch_size=int(args['--train-batch-size']),
                                           shuffle=True
                                           )
            #############################################################################
            # Define Model & Training Pipeline
            #############################################################################
            model = SpanEmo(output_dropout=float(args['--output-dropout']),
                            lang=args['--lang'],
                            joint_loss=args['--loss-type'],
                            alpha=float(args['--alpha-loss']))
            if case == "pretrained":
                tag_p = "" if lang == "English" else "DE"
                pretrained_model_dir = runs_ws / f"GoEmotionsSentiment{tag_p}"
                pretrained_name = sorted([model.name for model in pretrained_model_dir.glob("*.pt")])[0]
                pretrained_model_dir = str(runs_ws / f"GoEmotionsSentiment{tag_p}" / (pretrained_name))
                model.to(device).load_state_dict(torch.load(pretrained_model_dir))

            #############################################################################
            # Start Training
            #############################################################################
            cur_time = f"{localtime().tm_mday}.{localtime().tm_mon} - {localtime().tm_hour}:{localtime().tm_min}"
            run = name + " test - " + cur_time
            wandb.init(project=project, name=run)

            learn = Trainer(model, train_data_loader, dev_data_loader, "", args)
            learn.fit(
                num_epochs=int(args['--max-epoch']),
                args=args,
                device=device
            )

            wandb.finish()

            #############################################################################
            # Start Test
            #############################################################################
            model_name = sorted([model.name for model in model_dir.glob("*.pt")])[0]
            model_path = str(model_dir / (model_name))
            args['--test-path'] = args["--train-path"] = drive_ws / "SpanEmoData" / "E-c" / f"2018-E-c-En-test-gold{tag}.txt"
            args['--model-path'] = str(model_path)
            args['--test-batch-size'] = args['--train-batch-size']

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
            cur_time = f"{localtime().tm_mday}.{localtime().tm_mon} - {localtime().tm_hour}:{localtime().tm_min}"
            run = name + " test - " + cur_time
            wandb.init(project=project, name=run)

            model = SpanEmo(lang=args['--lang'])
            learn = EvaluateOnTest(model, test_data_loader, model_path=args['--model-path'])
            learn.predict(device=device)

            wandb.finish()
















