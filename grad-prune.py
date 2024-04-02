import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn
import math
import shutil

os.chdir(sys.path[0])
sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
from copy import deepcopy
import torch.nn.utils.prune as prune
import torchvision
import copy
from tqdm import tqdm

from defense.base import defense
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler
from utils.trainer_cls import ModelTrainerCLS_v2, BackdoorModelTrainer, Metric_Aggregator, given_dataloader_test, general_plot_for_epoch
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform, spc_choose_poisoned_sample
from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate, bd_attack_label_trans_generate
from utils.aggregate_block.dataset_and_transform_generate import get_dataset_normalization
from utils.trainer_cls import Metric_Aggregator, given_dataloader_test, plot_acc_like_metric, validate_list_for_plot

import matplotlib.pyplot as plt

class GradPrune_Dataset_Wrapper(torch.utils.data.Dataset):

    def __init__(self, clean_dataset, bd_dataset, end_img_transform, device):
        self.clean_dataset = clean_dataset
        self.bd_dataset = bd_dataset

        self.end_img_transform = end_img_transform
        self.device = device

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, index):
        img, label = self.clean_dataset[index]
        bd_img, bd_label = self.bd_dataset[index]

        img, bd_img = self.end_img_transform(img), self.end_img_transform(bd_img)

        return img, label, bd_img, bd_label

class GradPrune(defense):

    def __init__(self):
        super().__init__()
        pass

    def set_args(self, parser):

        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--model', type=str, help='resnet18')

        parser.add_argument("--dataset_path", type=str)
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, gtsrb, celeba, tiny')
        parser.add_argument("--num_classes", type=int)
        parser.add_argument("--input_height", type=int)
        parser.add_argument("--input_width", type=int)
        parser.add_argument("--input_channel", type=int)

        parser.add_argument('--yaml_path', type=str, default="./config/defense/grad_prune/config.yaml", help='the path of defense yaml')

        parser.add_argument('--attack', type=str)
        parser.add_argument('--patch_mask_path', type=str)
        parser.add_argument('--attack_label_trans', type=str, help='all2one, all2all, cleanLabel')
        parser.add_argument('--attack_target', type=int)

        parser.add_argument('--result_file', type=str, help='the location of result')
        parser.add_argument('--result_base', type=str, help='the location of result base path', default = "../record")
        parser.add_argument('--index', type=str, help='index of clean data')
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
        parser.add_argument('--spc', type=int, help='the samples per class used for training')
        parser.add_argument('--val_ratio', type=float, help='the ratio of validation data loader')
        parser.add_argument('--accuracy_threshold', type=float, help='the threshold of accuracy')
        parser.add_argument('--pruning_patience', type=int, help='the patience of loss when pruning')
        parser.add_argument('--tuning_patience', type=int, help='the patience of loss when tuning')

        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--amp', type=lambda x: str(x) in ['True', 'true', '1'])
        parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
        parser.add_argument('--batch_size', type=int)

        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
        parser.add_argument('--lr', type=float)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        
        # OMITTED FOR BREVITY
        # parser.add_argument('--sgd_momentum', type=float)
        # parser.add_argument('--wd', type=float, help='weight decay of sgd')
        # parser.add_argument('--client_optimizer', type=int)
        # parser.add_argument('--frequency_save', type=int,
        #                     help=' frequency_save, 0 is never')
        #parser.add_argument("-nb", "--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help=".to(), set the non_blocking = ?")

        
        # parser.add_argument('--epochs', type=int)
        # parser.add_argument('--lr', type=float)
        # parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')

        return parser
    
    def add_yaml_to_args(self, args):
        with open(args.yaml_path, 'r') as f:
            opt = yaml.safe_load(f)

        for key in opt:
            args.__dict__[key] = opt[key]

        return args
    
    def process_args(self, args):
        args.terminal_info = sys.argv
        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        save_path = args.result_base + os.path.sep + args.result_file + os.path.sep + "defense" + os.path.sep + "grad_prune" + os.path.sep + f"spc_{args.spc}" + os.path.sep + str(args.random_seed) + os.path.sep
        
        os.makedirs(save_path, exist_ok = True)
        args.save_path = save_path
        return args
    
    def prepare(self, args):

        ### set the logger
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()
        # file Handler
        fileHandler = logging.FileHandler(
            args.save_path + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel(logging.DEBUG)
        logger.addHandler(fileHandler)
        # consoleHandler
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        consoleHandler.setLevel(logging.INFO)
        logger.addHandler(consoleHandler)
        # overall logger level should <= min(handler) otherwise no log will be recorded.
        logger.setLevel(0)
        # disable other debug, since too many debug
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

        logging.info(pformat(args.__dict__))

        logging.debug("Only INFO or above level log will show in cmd. DEBUG level log only will show in log file.")

        # record the git infomation for debug (if available.)
        try:
            logging.debug(pformat(get_git_info()))
        except:
            logging.debug('Getting git info fails.')

        print(args.random_seed)
        fix_random(args.random_seed)
        self.args = args

        '''
                load_dict = {
                        'model_name': load_file['model_name'],
                        'model': load_file['model'],
                        'clean_train': clean_train_dataset_with_transform,
                        'clean_test' : clean_test_dataset_with_transform,
                        'bd_train': bd_train_dataset_with_transform,
                        'bd_test': bd_test_dataset_with_transform,
                    }
                '''
        self.attack_result = load_attack_result(args.result_base + os.path.sep + self.args.result_file + os.path.sep +'attack_result.pt')

        model = generate_cls_model(args.model, args.num_classes)
        model.load_state_dict(self.attack_result['model'])
        model.to(args.device)

        self.model = model
        attack_result = self.attack_result

        # Get the datasets
        clean_train_dataset = attack_result['clean_train']
        bd_train_dataset = attack_result['bd_train_all']

        end_img_transform = torchvision.transforms.Compose([
             torchvision.transforms.Resize((args.input_height, args.input_width)),
             torchvision.transforms.ToTensor(),
             get_dataset_normalization(args.dataset)
        ])

        clean_train_wrapper, clean_val_wrapper = copy.deepcopy(clean_train_dataset.wrapped_dataset), copy.deepcopy(clean_train_dataset.wrapped_dataset)
        clean_train_wrapper, clean_val_wrapper = prepro_cls_DatasetBD_v2(clean_train_wrapper), prepro_cls_DatasetBD_v2(clean_val_wrapper)
        bd_train_wrapper, bd_val_wrapper = bd_train_dataset.copy(), bd_train_dataset.copy()

        if args.spc is not None:
            train_idx, val_idx = spc_choose_poisoned_sample(bd_train_dataset, args.spc, args.val_ratio)
        else:
            ran_idx = choose_index(args, len(clean_train_wrapper))
            train_idx = np.random.choice(len(ran_idx), int(len(ran_idx) * (1-args.val_ratio)), replace=False)
            val_idx = np.setdiff1d(np.arange(len(ran_idx)), train_idx)

        clean_train_wrapper.subset(train_idx), clean_val_wrapper.subset(val_idx)
        bd_train_wrapper.subset(train_idx), bd_val_wrapper.subset(val_idx)

        print(f"Number of training samples: {len(clean_train_wrapper)}")
        print(f"Number of validation samples: {len(clean_val_wrapper)}")

        print(f"Number of BD training samples: {len(bd_train_wrapper)}")
        print(f"Number of BD validation samples: {len(bd_val_wrapper)}")

        clean_train_wrapper.getitem_all, clean_val_wrapper.getitem_all = False, False
        bd_train_wrapper.getitem_all, bd_val_wrapper.getitem_all = False, False

        train_dataset = GradPrune_Dataset_Wrapper(
            clean_train_wrapper,
            bd_train_wrapper,
            end_img_transform,
            args.device
        )

        val_dataset = GradPrune_Dataset_Wrapper(
            clean_val_wrapper,
            bd_val_wrapper,
            end_img_transform,
            args.device
        )

        #Show 5 samples of the dataset with and without the backdoor
        # fig, ax = plt.subplots(2, 5, figsize=(15, 5))
        # for i in range(5):

        #     img, label, bd_img, bd_label = train_dataset[i]

        #     ax[0, i].imshow(img.permute(1, 2, 0).cpu().numpy())
        #     ax[0, i].set_title(f'Clean Label: {label}')

        #     ax[1, i].imshow(bd_img.permute(1, 2, 0).cpu().numpy())
        #     ax[1, i].set_title(f'Backdoor Label: {bd_label}')

        # plt.show()

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )

        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )

        self.test_clean = torch.utils.data.DataLoader(
            attack_result['clean_test'],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )

        self.test_bd = torch.utils.data.DataLoader(
            attack_result['bd_test'],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )

    # -----------------------------------------------------------------------------------------------------------
    # Metrics Function

    def get_test_report_metrics(self, eval_model):

        clean_loss = 0
        rl_loss = 0

        clean_correct = 0
        bd_asr_correct = 0
        bd_ra_correct = 0

        for batch_idx, (x, labels) in tqdm(enumerate(self.test_clean)):

            x, labels = x.to(self.args.device), labels.to(self.args.device)
            eval_model.eval()
            with torch.no_grad():
                outputs = eval_model(x)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            clean_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            clean_correct += torch.sum(preds == labels.data)

        bd_pred = []
        bd_original = []
        bd_target = []

        for batch_idx, (x, labels, original_index, poison_indicator, original_targets) in tqdm(enumerate(self.test_bd)):

            x, labels, original_targets = x.to(self.args.device), labels.to(self.args.device), original_targets.to(self.args.device)
            eval_model.eval()
            with torch.no_grad():
                outputs = eval_model(x)
            loss = torch.nn.functional.cross_entropy(outputs, original_targets)
            rl_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            bd_pred.extend(preds.cpu().numpy())
            bd_target.extend(labels.cpu().numpy())
            bd_original.extend(original_targets.cpu().numpy())

        clean_accuracy = clean_correct.double() / len(self.test_clean.dataset)
        clean_accuracy = clean_accuracy.item()

        bd_asr_accuracy, bd_ra_accuracy = calculate_asr_ra(bd_pred, bd_original, bd_target)

        clean_loss = clean_loss / len(self.test_clean)
        rl_loss = rl_loss / len(self.test_bd)

        return clean_loss, rl_loss, clean_accuracy, bd_asr_accuracy, bd_ra_accuracy

    # -----------------------------------------------------------------------------------------------------------
    # Pruning Functions
    def prune_round_test(self, prune_model):

        rl_loss = 0
        clean_correct = 0

        for batch in self.val_loader:
            inputs, labels, inputs_b, _ = batch
            inputs, labels, inputs_b = inputs.to(self.args.device), labels.to(self.args.device), inputs_b.to(self.args.device)

            prune_model.eval()
            with torch.no_grad():
                outputs = prune_model(inputs)
                outputs_b = prune_model(inputs_b)

            _, preds = torch.max(outputs, 1)
            clean_correct += torch.sum(preds == labels.data)

            loss = torch.nn.functional.cross_entropy(outputs_b, labels)
            rl_loss += loss.item()

        clean_accuracy = clean_correct.double() / len(self.val_loader.dataset)
        clean_accuracy = clean_accuracy.item()

        return clean_accuracy, rl_loss

    def grad_prune(self, prune_model):

        prune_model.zero_grad()
        prune_model.train()

        loss_rl = 0
        loss_clean = 0

        clean_correct = 0
        clean_total = 0
        pred_bd = []
        original = []
        target = []

        # Get gradients for all training samples
        for batch in tqdm(self.train_loader):
            inputs, labels, inputs_b, labels_b = (x.to(self.args.device) for x in batch)
            logits_b = prune_model(inputs_b)
            _, preds = torch.max(logits_b, 1)

            pred_bd.extend(preds.cpu().numpy())
            original.extend(labels.cpu().numpy())
            target.extend(labels_b.cpu().numpy())

            loss = torch.nn.functional.cross_entropy(logits_b, labels)
            loss_rl += loss.item()

            loss.backward()

            prune_model.eval()
            with torch.no_grad():
                logits = prune_model(inputs)

            _, preds = torch.max(logits, 1)
            clean_correct += torch.sum(preds == labels.data)
            clean_total += len(labels)

            loss_clean += torch.nn.functional.cross_entropy(logits, labels).item()

        # Find the filter with the largest absolute mean gradient across all parameters
        best_value = 0
        best_index = None

        all_params = list(prune_model.parameters())
        for layer_index, param in enumerate(all_params):

            if param.requires_grad:
                if len(param.grad.shape) == 4:
                    mean_grad = get_mean_layer(param.grad)

                    for filter_index, value in enumerate(mean_grad):
                        if value > best_value:
                            best_value = value
                            best_index = (layer_index, filter_index)

        # Set the filter and bias to zero
        layer_index, filter_index = best_index
        all_params[layer_index].data[filter_index] = torch.zeros(all_params[layer_index].data[filter_index].shape)
        all_params[layer_index + 1].data[filter_index] = torch.zeros(all_params[layer_index + 1].data[filter_index].shape) 

        acc = clean_correct.double() / clean_total
        acc = acc.item()

        asr, ra = calculate_asr_ra(pred_bd, original, target)

        loss_clean = loss_clean / len(self.train_loader)
        loss_rl = loss_rl / len(self.train_loader)

        return loss_clean, loss_rl, acc, asr, ra
        #return loss_clean, loss_rl, acc, asr, ra

    def iterative_prune(self):

        prune_metric = Metric_Aggregator()

        # Setup the lists to store the metrics
        train_loss_clean, train_loss_rl = [], []
        train_acc, train_asr, train_rl = [], [], []

        test_loss_clean, test_loss_rl = [], []
        test_acc, test_asr, test_ra = [], [], []

        # Copy the model to ensure the original model is not modified
        prune_model = copy.deepcopy(self.model)

        # Get the initial validation metrics - Save the initial model to ensure the best model is saved
        val_clean_accuracy, val_rl_loss = self.prune_round_test(prune_model)
        #print(f'Validation: Initial Accuracy: {val_clean_accuracy:.4f}, Initial Backdoor Loss: {val_rl_loss:.4f}')

        val_best_loss = val_rl_loss
        val_initial_accuracy = val_clean_accuracy
        best_prune_model = copy.deepcopy(prune_model)
        patience = 0

        # Prune the model until the accuracy threshold is reached or the patience is exceeded
        while val_initial_accuracy - val_clean_accuracy < self.args.accuracy_threshold and patience < self.args.pruning_patience:

            # Test the initial model
            round_test_clean_loss, round_test_rl_loss, round_test_clean_accuracy, round_test_bd_asr_accuracy, round_test_bd_ra_accuracy = self.get_test_report_metrics(prune_model)
            test_loss_clean.append(round_test_clean_loss), test_loss_rl.append(round_test_rl_loss), test_acc.append(round_test_clean_accuracy), test_asr.append(round_test_bd_asr_accuracy), test_ra.append(round_test_bd_ra_accuracy)

            # Prune the model and get the metrics
            round_train_clean_loss, round_train_rl_loss, round_train_clean_accuracy, round_train_bd_asr_accuracy, round_train_bd_ra_accuracy = self.grad_prune(prune_model)
            train_loss_clean.append(round_train_clean_loss), train_loss_rl.append(round_train_rl_loss), train_acc.append(round_train_clean_accuracy), train_asr.append(round_train_bd_asr_accuracy), train_rl.append(round_train_bd_ra_accuracy)

            # Test the model using the validation set
            val_clean_accuracy, val_bd_loss = self.prune_round_test(prune_model)
            val_clean_delta = val_initial_accuracy - val_clean_accuracy

            # Save the metrics - These are added here to ensure the metrics are saved even if the model is not saved
            # Note: First epoch uses the initial test metrics
            prune_metric({
                'train_clean_loss': round_train_clean_loss,
                'train_rl_loss': round_train_rl_loss,
                'train_clean_accuracy': round_train_clean_accuracy,
                'train_asr_accuracy': round_train_bd_asr_accuracy,
                'train_rl_accuracy': round_train_bd_ra_accuracy,
                'test_clean_loss': round_test_clean_loss,
                'test_rl_loss': round_test_rl_loss,
                'test_clean_accuracy': round_test_clean_accuracy,
                'test_asr_accuracy': round_test_bd_asr_accuracy,
                'test_ra_accuracy': round_test_bd_ra_accuracy
            })

            plot_loss_grad_prune(
                train_loss_clean, train_loss_rl,
                test_loss_clean, test_loss_rl,
                self.args.save_path,
                save_file_name="loss_metric_plots_prune"
            )

            plot_acc_like_metric(
                train_acc, train_asr, train_rl,
                test_acc, test_asr, test_ra,
                self.args.save_path,
                save_file_name="acc_metric_plots_prune"
            )

            prune_metric.to_dataframe().to_csv(self.args.save_path + 'prune_metrics.csv')

            # Save the model if it is the best model
            if val_bd_loss < val_best_loss and val_clean_delta < self.args.accuracy_threshold:
                val_best_loss = val_bd_loss
                best_prune_model = copy.deepcopy(prune_model)
                patience = 0
            else:
                patience += 1

        # Set the current model to the best model
        self.model = best_prune_model
        prune_metric.to_dataframe().to_csv(self.args.save_path + 'prune_metrics.csv')

    # -----------------------------------------------------------------------------------------------------------
    # Fine-Tuning Method
    def tuning_round_test(self, tune_model):

        clean_loss = 0

        for batch in self.val_loader:
            inputs, labels, _, _ = batch
            inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)

            tune_model.eval()
            with torch.no_grad():
                outputs = tune_model(inputs)

            loss = torch.nn.functional.cross_entropy(outputs, labels)
            clean_loss += loss.item()

        return clean_loss

    def fine_tuning(self):

        tune_metric = Metric_Aggregator()

        # Setup the lists to store the metrics
        train_loss_clean, train_loss_rl = [], []
        train_acc, train_asr, train_rl = [], [], []

        test_loss_clean, test_loss_rl = [], []
        test_acc, test_asr, test_ra = [], [], []

        # Get the initial validation metrics - Save the initial model to ensure the best model is saved
        tuning_model = copy.deepcopy(self.model)
        optimizer, _ = argparser_opt_scheduler(tuning_model, self.args)

        val_best_loss = self.tuning_round_test(tuning_model)
        best_tuning_model = copy.deepcopy(tuning_model)
        patience = 0

        while patience < self.args.tuning_patience:

            # Test the current model
            round_test_clean_loss, round_test_rl_loss, round_test_clean_accuracy, round_test_bd_asr_accuracy, round_test_bd_ra_accuracy = self.get_test_report_metrics(tuning_model)
            test_loss_clean.append(round_test_clean_loss), test_loss_rl.append(round_test_rl_loss), test_acc.append(round_test_clean_accuracy), test_asr.append(round_test_bd_asr_accuracy), test_ra.append(round_test_bd_ra_accuracy)

            # Setup training metrics
            round_train_loss_clean, round_train_loss_rl = 0, 0
            clean_correct = 0
            clean_total = 0

            pred_bd = []
            original = []
            target = []

            tuning_model.train()
            tuning_model.zero_grad()
            for batch in tqdm(self.train_loader):
                optimizer.zero_grad()

                # Get the batch
                inputs, labels, inputs_b, labels_b = batch
                inputs, labels, inputs_b, labels_b = inputs.to(self.args.device), labels.to(self.args.device), inputs_b.to(self.args.device), labels_b.to(self.args.device)

                # Forward pass - Clean inputs
                outputs = tuning_model(inputs)
                _, preds = torch.max(outputs, 1)

                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()

                round_train_loss_clean += loss.item()
                clean_correct += torch.sum(preds == labels.data)
                clean_total += len(labels)

                # Forward pass - Backdoor inputs
                outputs = tuning_model(inputs_b)
                _, preds = torch.max(outputs, 1)

                loss_b = torch.nn.functional.cross_entropy(outputs, labels)
                loss_b.backward()

                round_train_loss_rl += loss_b.item()
                pred_bd.extend(preds.cpu().numpy())
                original.extend(labels.cpu().numpy())
                target.extend(labels_b.cpu().numpy())

                optimizer.step()

            # Calculate the training metrics
            round_train_acc = clean_correct.double() / clean_total
            round_train_acc = round_train_acc.item()
            round_train_asr, round_train_rl = calculate_asr_ra(pred_bd, original, target)

            round_train_loss_clean = round_train_loss_clean / len(self.train_loader)
            round_train_loss_rl = round_train_loss_rl / len(self.train_loader)

            train_loss_clean.append(round_train_loss_clean), train_loss_rl.append(round_train_loss_rl), train_acc.append(round_train_acc), train_asr.append(round_train_asr), train_rl.append(round_train_rl)

            # Save the metrics
            tune_metric({
                'train_clean_loss': round_train_loss_clean,
                'train_rl_loss': round_train_loss_rl,
                'train_clean_accuracy': round_train_acc,
                'train_asr_accuracy': round_train_asr,
                'train_rl_accuracy': round_train_rl,
                'test_clean_loss': round_test_clean_loss,
                'test_rl_loss': round_test_rl_loss,
                'test_clean_accuracy': round_test_clean_accuracy,
                'test_asr_accuracy': round_test_bd_asr_accuracy,
                'test_rl_accuracy': round_test_bd_ra_accuracy
            })

            # Plot the metrics
            plot_loss_grad_prune(
                train_loss_clean, train_loss_rl,
                test_loss_clean, test_loss_rl,
                self.args.save_path,
                save_file_name="loss_metric_plots_tune"
            )

            plot_acc_like_metric(
                train_acc, train_asr, train_rl,
                test_acc, test_asr, test_ra,
                self.args.save_path,
                save_file_name="acc_metric_plots_tune"
            )

            tune_metric.to_dataframe().to_csv(self.args.save_path + 'tune_metrics.csv')

            # Test the model using the validation set
            val_clean_loss = self.tuning_round_test(tuning_model)

            if val_clean_loss < val_best_loss:
                val_best_loss = val_clean_loss
                best_tuning_model = copy.deepcopy(tuning_model)
                patience = 0

            else:
                patience += 1

        # Save the model
        self.model = best_tuning_model
        tune_metric.to_dataframe().to_csv(self.args.save_path + 'tune_metrics.csv')

    def defense(self):

        self.iterative_prune()
        self.fine_tuning()

        result = {}
        result['model'] = self.model
        save_defense_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=self.model.cpu().state_dict(),
            save_path=args.save_path,
        )

# -----------------------------------------------------------------------------------------------------------
# Utility Functions
def get_mean_layer(grads):

    filter_means = []
    num_filters = grads.shape[0]

    for i in range(num_filters):
        filter_grad = grads[i]       
        filter_grad_abs = torch.abs(filter_grad)

        filter_mean = torch.mean(filter_grad_abs)
        filter_means.append(filter_mean)
        
    return filter_means

def calculate_asr_ra(predicted_labels, original_labels, target_labels):

    asr_correct = 0
    ra_correct = 0

    for i in range(len(predicted_labels)):

        if predicted_labels[i] == original_labels[i]:
            ra_correct += 1

        if predicted_labels[i] == target_labels[i] and predicted_labels[i] != original_labels[i]:
            asr_correct += 1

    asr = asr_correct / len(predicted_labels)
    ra = ra_correct / len(predicted_labels)

    return asr, ra

# Note: This function is a modified version of the plot_loss function from trainer_cls.py to allow for the plotting of the RL loss (Relearning Loss)
def plot_loss_grad_prune(
        train_loss_clean_list : list,
        train_loss_rl_list : list,
        test_loss_clean_list : list,
        test_loss_rl_list : list,
        save_folder_path: str,
        save_file_name="loss_metric_plots",
    ):
    '''These line of set color is from https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib'''
    NUM_COLORS = 4
    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure(figsize=(12.8, 9.6)) #  4x default figsize
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

    len_set = len(train_loss_clean_list)
    x = range(len_set)
    if validate_list_for_plot(train_loss_clean_list, len_set):
        plt.plot(x, train_loss_clean_list, marker="o", linewidth=2, label="Train Loss (Clean)", linestyle="--")
    else:
        logging.warning("train_loss_clean_list contains None or len not match")

    if validate_list_for_plot(train_loss_rl_list, len_set):
        plt.plot(x, train_loss_rl_list, marker="x", linewidth=2, label="Train Loss (Relearning)", linestyle="-.")
    else:
        logging.warning("train_loss_rl_list contains None or len not match")

    if validate_list_for_plot(test_loss_clean_list, len_set):
        plt.plot(x, test_loss_clean_list, marker="*", linewidth=2, label="Test Loss (Clean)", linestyle="-")
    else:
        logging.warning("test_loss_clean_list contains None or len not match")

    if validate_list_for_plot(test_loss_rl_list, len_set):
        plt.plot(x, test_loss_rl_list, marker="v", linewidth=2, label="Test Loss (Relearning)", linestyle=":")
    else:
        logging.warning("bd_test_loss_list contains None or len not match")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.ylim((0,
        max([value for value in  # filter None value
            train_loss_clean_list + train_loss_rl_list + test_loss_clean_list + test_loss_rl_list if value is not None]
        )))
    plt.legend()
    plt.title("Results")
    plt.grid()
    plt.savefig(f"{save_folder_path}/{save_file_name}.png")
    plt.close()


if __name__ == '__main__':
    grad_prune = GradPrune()

    parser = argparse.ArgumentParser()
    parser = grad_prune.set_args(parser)
    args = parser.parse_args()

    grad_prune.add_yaml_to_args(args)
    args = grad_prune.process_args(args)

    grad_prune.prepare(args)
    grad_prune.defense()