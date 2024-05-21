# Créé le générateur de dataset, puis le fait générer en se basant sur le fichier list_of_cheese

import torch
import wandb
import hydra
from tqdm import tqdm


@hydra.main(config_path="configs/generate", config_name="config")
def generate(cfg):
    dataset_generator = hydra.utils.instantiate(cfg.dataset_generator) #pointe vers le dataset_generator de config.yaml qui pointe vers first_week

    with open(cfg.labels_file, "r") as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]

    dataset_generator.generate(labels)


if __name__ == "__main__":
    generate()
