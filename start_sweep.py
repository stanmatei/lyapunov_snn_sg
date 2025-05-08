import yaml 
import argparse
import wandb
import os
import contextlib


def load_yaml_to_dict(file_path):
    """
    Loads a YAML file and returns its content as a Python dictionary.
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except yaml.YAMLError as e:
         print(f"Error parsing YAML file: {e}")
         return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str)
    parser.add_argument("--project", type=str)
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    file_path = args.config
    sweep_config = load_yaml_to_dict(file_path)
    
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        sweep_id = wandb.sweep(sweep=sweep_config, entity=args.entity, project=args.project)
    print(args.entity + "/" + args.project + "/" + sweep_id)