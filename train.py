import argparse
import logging
import pickle
import os

import xgboost as xgb
from dvclive import Live
from dvclive.xgb import DVCLiveCallback


if __name__ == '__main__':
    logging.info("starting train script.")
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here
    parser.add_argument("--max_depth", type=int)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--gamma", type=int)
    parser.add_argument("--min_child_weight", type=int)
    parser.add_argument("--subsample", type=float)
    parser.add_argument("--silent", type=int)
    parser.add_argument("--objective", type=str, default="binary:logistic")
    parser.add_argument("--num_round", type=int)

    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))

    args = parser.parse_args()

    train_hp = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'gamma': args.gamma,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'silent': args.silent,
        #'objective': args.objective,
    }

    dtrain = xgb.DMatrix(args.train + "?format=csv&label_column=0")
    dval = xgb.DMatrix(args.validation + "?format=csv&label_column=0")
    watchlist = [(dtrain, 'train'), (dval, 'validation')] if dval is not None else [(dtrain, 'train')]

    with Live(os.path.join(args.model_dir + "/dvclive")) as live:
        live.log_param("cwd", os.getcwd())
    
        callbacks = [DVCLiveCallback()]

        bst = xgb.train(
            params=train_hp,
            dtrain=dtrain,
            evals=watchlist,
            num_boost_round=args.num_round,
            callbacks=callbacks
        )

    # Save the model to the location specified by ``model_dir``
    model_location = args.model_dir + '/xgboost-model'
    pickle.dump(bst, open(model_location, 'wb'))
    logging.info("Stored trained model at {}".format(model_location))