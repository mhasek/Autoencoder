import numpy as np                                                                                    
import tensorflow as tf

from trainer import Trainer
from config import get_config
from data_loader import get_loader
from utils import prepare_dirs_and_logger, save_config

def main(config):
  prepare_dirs_and_logger(config)

  rng = np.random.RandomState(config.random_seed)
  tf.set_random_seed(config.random_seed)

  train_data_loader = get_loader(
    config.data_path, config.batch_size, True)


  trainer = Trainer(config, train_data_loader)
  if config.is_train:
    save_config(config)
    trainer.train()
  else:
    if not config.load_path:
      raise Exception("[!] You should specify `load_path` to load a pretrained model")
    trainer.gen_image()

if __name__ == "__main__":
  config, unparsed = get_config()
  main(config)
