import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Utils import load
from Utils import generator
from Utils import metrics
from train import *
from prune import *
from logging_code import logger
import os



from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        if 'bias' in name:
            continue

        param = parameter.numel()
        param_zero = (parameter ==0).sum()
        param = param - param_zero
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
def run(args):

    logger.setup_logger(args)

    if not args.save:
        print("This experiment requires an expid.")
        quit()

    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load.device(args.gpu)
    print(device)
    logger.print_and_log(device)
    ## Data ##
    print('Loading {} dataset.'.format(args.dataset))
    logger.print_and_log('Loading {} dataset.'.format(args.dataset))
    input_shape, num_classes = load.dimension(args.dataset) 
    prune_loader = load.dataloader(args.dataset, args.prune_batch_size, True, args.workers, args.prune_dataset_ratio * num_classes)
    train_loader = load.dataloader(args.dataset, args.train_batch_size, True, args.workers)
    test_loader = load.dataloader(args.dataset, args.test_batch_size, False, args.workers)

    ## Model ##
    print('Creating {} model.'.format(args.model))
    logger.print_and_log('Creating {} model.'.format(args.model))
    model = load.model(args.model, args.model_class)(input_shape, 
                                                     num_classes, 
                                                     args.dense_classifier,
                                                     args.pretrained).to(device)
    loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = load.optimizer(args.optimizer)
    optimizer = opt_class(generator.parameters(model), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)
    def print_param(model):
        for param in model.parameters():
            print(param)
            break
    ## Save Original ##
    torch.save(model.state_dict(),"{}/model.pt".format(args.result_dir))
    torch.save(optimizer.state_dict(),"{}/optimizer.pt".format(args.result_dir))
    torch.save(scheduler.state_dict(),"{}/scheduler.pt".format(args.result_dir))

    
    total_params_count_trainable = count_parameters(model)
    ## Train-Prune Loop ##

    for compression in args.compression_list:
        
        for level in args.level_list:

            print('{} compression ratio, {} train-prune levels'.format(compression, level))
            logger.print_and_log('{} compression ratio, {} train-prune levels'.format(compression, level))
            # Reset Model, Optimizer, and Scheduler
            model.load_state_dict(torch.load("{}/model.pt".format(args.result_dir), map_location=device))
            optimizer.load_state_dict(torch.load("{}/optimizer.pt".format(args.result_dir), map_location=device))
            scheduler.load_state_dict(torch.load("{}/scheduler.pt".format(args.result_dir), map_location=device))

            sparsities = []
            sparsity = 0
            
            for l in range(level):
                

                # Pre Train Model
                train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                test_loader, device, args.pre_epochs, args.verbose,l,sparsity)

                

                # Prune Model
                pruner = load.pruner(args.pruner)(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
                sparsity = (10**(-float(compression)))**((l + 1) / level)
                sparsities.append(sparsity)
                
                #eval_aft_level_train(model, loss, test_loader, device, args.verbose,l,sparsity)

                if l !=0:

                 torch.save(model.state_dict(),"{}/sparisty{}_model.pt".format(args.result_dir,sparsities[l-1]))

                remaining_params, total_params = prune_loop(model, loss, pruner, prune_loader, device, sparsity,
                           args.compression_schedule, args.mask_scope, args.prune_epochs, args.reinitialize, args.prune_train_mode, args.shuffle, args.invert,args.result_dir)

                # Reset Model's Weights
                original_dict = torch.load("{}/model.pt".format(args.result_dir), map_location=device)
                original_weights = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), original_dict.items()))
                model_dict = model.state_dict()
                model_dict.update(original_weights)
                model.load_state_dict(model_dict)
                
                # Reset Optimizer and Scheduler
                optimizer.load_state_dict(torch.load("{}/optimizer.pt".format(args.result_dir), map_location=device))
                scheduler.load_state_dict(torch.load("{}/scheduler.pt".format(args.result_dir), map_location=device))
                prune_apply(model,pruner)
                # prune_result = metrics.summary(model, 
                #                            pruner.scores,
                #                            metrics.flop(model, input_shape, device),
                #                            lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual))
                total_params_count_trainable_remaining = count_parameters(model)
                if (remaining_params == total_params_count_trainable_remaining):
                        print(str(remaining_params)+"are_pruned till now")
                        logger.print_and_log(str(remaining_params)+"are_pruned till now")
            # Prune Result
            prune_result = metrics.summary(model, 
                                           pruner.scores,
                                           metrics.flop(model, input_shape, device),
                                           lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual))
            # Train Model
            post_result = post_prune_train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                          test_loader, device, args.post_epochs, args.verbose,l+1,sparsities[l],args.result_dir)
            total_params_count_trainable_remaining = count_parameters(model)
            print("at last the remaning params are"+str(total_params_count_trainable_remaining))
            logger.print_and_log("at last the remaning params are"+str(total_params_count_trainable_remaining))
            # Save Data
            os.remove("{}/prev_mask.pt".format(args.result_dir))
            torch.save(model.state_dict(),"{}/post_train_model.pt".format(args.result_dir, args.pruner, str(compression),  str(level)))
            #post_result.to_pickle("{}/post-train-{}-{}-{}.pkl".format(args.result_dir, args.pruner, str(compression),  str(level)))
            prune_result.to_pickle("{}/compression-{}-{}-{}.pkl".format(args.result_dir, args.pruner, str(compression), str(level)))


