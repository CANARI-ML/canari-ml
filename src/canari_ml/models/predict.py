import datetime as dt
import importlib
import logging
import os

import lightning.pytorch as pl
import numpy as np
import torch

from ..data.dataloader import CANARIMLDataSetTorch
from .cli import predict_args


def load_module(module_path, module_name, module_args):
    module = importlib.import_module(module_path)
    lightning_module_class = getattr(module, module_name)
    lightning_module = lightning_module_class(**module_args)
    return lightning_module


def predict_forecast(
    dataset_config: object,
    network_name: object,
    dataset_name: object = None,
    network_folder: object = None,
    output_folder: object = None,
    save_args: bool = False,
    seed: int = 42,
    start_dates: object = tuple([dt.datetime.now().date()]),
    test_set: bool = False,
) -> object:
    # TODO: going to need to be able to handle merged datasets
    ds = CANARIMLDataSetTorch(dataset_config)
    dl = ds.get_data_loader()

    if not network_folder:
        network_folder = os.path.join(".", "results", "networks", network_name)

    dataset_name = dataset_name if dataset_name else ds.identifier
    model_path = os.path.join(
        network_folder, "{}.model_{}.{}.ckpt".format(network_name, dataset_name, seed)
    )

    logging.info("Loading model from {}...".format(model_path))

    torch.serialization.add_safe_globals(
        [
            "lightning_module_name",
            "lightning_module_path",
            "hyper_parameters",
            "state_dict",
        ]
    )
    checkpoint = torch.load(
        model_path, map_location=torch.device("cpu"), weights_only=False
    )
    lightning_module_name = checkpoint["lightning_module_name"]
    lightning_module_path = checkpoint["lightning_module_path"]
    lightning_module_args = checkpoint["hyper_parameters"]

    if not lightning_module_name:
        raise ValueError("Checkpoint is missing lightning module class name")

    lightning_module = load_module(
        lightning_module_path, lightning_module_name, lightning_module_args
    ).eval()

    lightning_module.load_state_dict(checkpoint["state_dict"])
    lightning_module.eval()
    # lightning_module.load_from_checkpoint(model_path)

    if not test_set:
        logging.info("Generating forecast inputs from processed/ files")
        for date in start_dates:
            data_sample = dl.generate_sample(date, prediction=True)

            input_sample = torch.tensor(data_sample[0]).unsqueeze(dim=0)

            logging.info("Running prediction {}".format(date))
            with torch.no_grad():
                predictions = lightning_module(input_sample).unsqueeze(dim=0)

            save_prediction(
                predictions=predictions,
                dates=[date],
                output_folder=output_folder,
            )
    else:
        logging.info("Using forecast inputs from network_dataset/ files")

        _, _, test_inputs = ds.get_data_loaders(ratio=1.0)
        # itval = iter(test_inputs)
        # nxt = next(itval)
        # print(nxt[0].shape)

        trainer = pl.Trainer()
        with torch.no_grad():
            predictions = trainer.predict(lightning_module, dataloaders=test_inputs)

        source_key = [k for k in dl.config["sources"].keys() if k != "meta"][0]

        test_dates = [
            dt.date(*[int(v) for v in d.split("_")])
            for d in dl.config["sources"][source_key]["dates"]["test"]
        ]

        save_prediction(
            predictions=predictions,
            dates=test_dates,
            output_folder=output_folder,
        )


def save_prediction(predictions, dates, output_folder):
    if os.path.exists(output_folder):
        logging.warning("{} output already exists".format(output_folder))
    os.makedirs(output_folder, exist_ok=output_folder)

    idx = 0
    for workers, prediction in enumerate(predictions):
        for batch in range(prediction.shape[0]):
            date = dates[idx]
            logging.info(
                "Saving {} - forecast output {}".format(date, prediction.shape)
            )
            output_path = os.path.join(output_folder, date.strftime("%Y_%m_%d.npy"))
            forecast = prediction[batch, :, :, :, :].movedim(-2, 0)
            forecast_np = forecast.detach().cpu().numpy()
            np.save(output_path, forecast_np)
            idx += 1

    return output_path


def main():
    args = predict_args()

    dataset_config = os.path.join(".", "dataset_config.{}.json".format(args.dataset))

    date_content = args.datefile.read()
    dates = [dt.date(*[int(v) for v in s.split("-")]) for s in date_content.split()]
    args.datefile.close()

    output_folder = os.path.join(
        ".",
        "results",
        "predict",
        args.output_name,
        "{}.{}".format(args.network_name, args.seed),
    )

    logging.info("Prediction output directory:", output_folder)

    predict_forecast(
        dataset_config,
        args.network_name,
        dataset_name=args.ident if args.ident else args.dataset,
        output_folder=output_folder,
        save_args=args.save_args,
        seed=args.seed,
        start_dates=dates,
        test_set=args.testset,
    )


if __name__ == "__main__":
    # import torch.multiprocessing as mp

    # mp.set_start_method("spawn")
    main()
