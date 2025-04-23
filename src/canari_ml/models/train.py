import json
import logging
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp

from ..data.dataloader import CANARIMLDataSetTorch
from . import losses, metrics
from .cli import TrainingArgParser
from .models import LitUNet, unet_batchnorm
from .networks.pytorch import PytorchNetwork


def evaluate_model(
    trainer: object, model_path: object, dataset: object, dataset_ratio: float = 1.0
):
    logging.info("Running evaluation against test set")

    # _, val_ds, test_ds = dataset.get_split_datasets(ratio=dataset_ratio)
    _, validation_dataloader, test_dataloader = dataset.get_data_loaders(
        ratio=dataset_ratio
    )
    eval_data = validation_dataloader

    if dataset.counts["test"] > 0:
        eval_data = test_dataloader
        logging.info("Using test set for validation")
    else:
        logging.warning(
            "Using validation data source for evaluation, rather than test set"
        )

    lead_times = list(range(1, dataset.lead_time + 1))
    logging.info("Metric creation for lead time of {} days".format(len(lead_times)))
    # # TODO: common across train_model and evaluate_model - list of instantiations
    # metric_names = ["binacc", "mae", "rmse"]
    # metrics_classes = [
    #     metrics.WeightedBinaryAccuracy,
    #     metrics.WeightedMAE,
    #     metrics.WeightedRMSE,
    # ]
    # metrics_list = [
    #     cls(leadtime_idx=lt - 1) for lt in lead_times
    #     for cls in metrics_classes
    # ]

    # network.compile(weighted_metrics=metrics_list)

    logging.info("Evaluating... ")
    tic = time.time()

    # Load the best result from the checkpoint
    best_model = LitUNet.load_from_checkpoint(model_path)

    # disable randomness, dropout, etc...
    best_model.eval()

    with torch.no_grad():
        results = trainer.test(best_model, dataloaders=eval_data)
    print(results)

    # results = network.evaluate(
    #     eval_data,
    #     return_dict=True,
    #     verbose=2
    # )

    output_path = Path(model_path).resolve().with_suffix("")
    results_path = "{}.results.json".format(output_path)
    logging.info(f"Saving evaluation results to {results_path}")
    with open(results_path, "w") as fh:
        json.dump(results[0], fh)

    logging.debug(results)
    logging.info("Done in {:.1f}s".format(time.time() - tic))

    return results
    # return results, metric_names, lead_times


def get_datasets_torch(args):
    dataset_filenames = [
        el if str(el).split(".")[-1] == "json" else "dataset_config.{}.json".format(el)
        for el in [args.dataset, *args.additional]
    ]

    # if len(args.additional) == 0:
    #     dataset = IceNetDataSetPyTorch(dataset_filenames[0],
    #                                     batch_size=args.batch_size,
    #                                     shuffling=args.shuffle_train)
    # else:
    #     dataset = MergedIceNetDataSet(dataset_filenames,
    #                                   batch_size=args.batch_size,
    #                                   shuffling=args.shuffle_train)

    dataset = CANARIMLDataSetTorch(
        dataset_filenames[0], batch_size=args.batch_size, shuffling=args.shuffle_train
    )
    return dataset


def execute_pytorch_training(args, dataset, network, save=True, evaluate=True):
    # There is a better way of doing this by passing off to a dynamic factory
    # for other integrations, but for the moment I have no shame
    using_wandb = False
    run = None

    # # TODO: move to overridden implementation - decorator?
    # if not args.no_wandb:
    #     from icenet.model.handlers.wandb import init_wandb, finalise_wandb
    #     run, callback = init_wandb(args)

    #     if callback is not None:
    #         network.add_callback(callback)
    #         using_wandb = True

    input_shape = (*dataset.shape, dataset.num_channels) # 720, 720, 4
    ratio = args.ratio if args.ratio else 1.0
    # Do not yet support ratio != 1.0 with pytorch
    train_dataloader, validation_dataloader, _ = dataset.get_data_loaders(ratio=1.0)
    # train_ds, val_ds, _ = dataset.get_split_datasets(ratio=ratio)

    trainer, checkpoint_callback = network.train(
        args.epochs,
        unet_batchnorm,
        train_dataloader,
        model_creator_kwargs=dict(
            input_shape=input_shape,
            loss=losses.MSELoss(),
            #loss=losses.HuberLoss(),
            # Note, when using CLI, pass the metric method name prepended by 'val_'
            # e.g. `--checkpoint-monitor val_mse`
            metrics=[
                metrics.MAE,
                metrics.MSE,
                metrics.RMSE,
                # losses.HuberLoss,
                # losses.MSELoss,
            ],
            learning_rate=args.lr,
            filter_size=args.filter_size,
            n_filters_factor=args.n_filters_factor,
            lead_time=dataset.lead_time,
        ),
        save=save,
        validation_dataloader=validation_dataloader,
    )

    if evaluate:
        best_checkpoint = checkpoint_callback.best_model_path
        logging.info(f"Best checkpoint: {best_checkpoint}")
        # results, metric_names, leads = \
        #     evaluate_model(best_checkpoint,
        #                    dataset,
        #                    dataset_ratio=args.ratio)
        results = evaluate_model(
            trainer, best_checkpoint, dataset, dataset_ratio=args.ratio
        )

    #     if using_wandb:
    #         finalise_wandb(run, results, metric_names, leads)


def pytorch_main():
    # mp.set_start_method("spawn")
    args = TrainingArgParser().add_unet().parse_args()
    dataset = get_datasets_torch(args)
    network = PytorchNetwork(
        dataset,
        args.run_name,
        checkpoint_mode=args.checkpoint_mode,
        checkpoint_monitor=args.checkpoint_monitor,
        # early_stopping_patience=args.early_stopping,
        # lr_decay=(
        #     args.lr_10e_decay_fac,
        #     args.lr_decay_start,
        #     args.lr_decay_end,
        # ),
        pre_load_path=args.preload,
        seed=args.seed,
        verbose=args.verbose,
    )
    execute_pytorch_training(args, dataset, network)


if __name__ == "__main__":
    # mp.set_start_method("spawn")
    pytorch_main()
