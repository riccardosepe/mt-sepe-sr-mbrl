"""
Save tensorboard scalars to a JSON file.
"""
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from argparse import ArgumentParser
import json


def prepare_folders(in_folder):
    if not has_subfolders(in_folder):
        raise UserWarning(f"No subfolders found in {in_folder}")

    base_path = os.path.join(in_folder, "tb")
    os.mkdir(base_path)

    for folder in os.listdir(in_folder):
        if folder == "tb":
            continue
        folder_path = os.path.join(in_folder, folder)
        if not os.path.isdir(folder_path):
            continue

        new_path = os.path.join(base_path, folder)
        tb_path = os.path.join(folder_path, "tensorboard")
        os.system(f"mv {tb_path} {base_path}")

        tmp_path = os.path.join(base_path, "tensorboard")
        os.system(f"mv {tmp_path} {new_path}")

    return base_path


def has_subfolders(folder):
    return any(os.path.isdir(os.path.join(folder, f)) for f in os.listdir(folder))


def process_run(run_folder):
    event_acc = EventAccumulator(run_folder)
    event_acc.Reload()

    scalars = event_acc.Tags()['scalars']
    if not scalars or len(scalars) == 0:
        print(f"No scalar data found in {run_folder}")
        return None

    run_data = {}

    for scalar in scalars:
        steps = []
        values = []
        data = event_acc.Scalars(scalar)
        for d in data:
            steps.append(int(d.step))
            values.append(d.value)
        run_data[scalar] = {'steps': steps, 'values': values}

    return run_data


def main(input_folder, output_folder):
    all_data = {}  # Dictionary to hold data from all runs

    if not has_subfolders(input_folder):
        run_data = process_run(input_folder)
        if run_data is not None:
            all_data[os.path.basename(input_folder)] = run_data
        else:
            raise ValueError(f"No scalar data found in {input_folder}")

    # Iterate through each subfolder (run)
    for run_folder in os.listdir(input_folder):
        run_path = os.path.join(input_folder, run_folder)
        if not os.path.isdir(run_path):
            continue  # Skip if not a directory

        run_data = process_run(run_path)
        if run_data is not None:
            all_data[run_folder] = run_data

    output_file = os.path.join(output_folder, 'tensorboard.json')
    i = 1
    while os.path.exists(output_file):
        output_file = os.path.join(output_folder, f'tensorboard_{i}.json')
        i += 1

    with open(output_file, 'w') as f:
        json.dump(all_data, f)

    return output_file


if __name__ == '__main__':
    arg_parser = ArgumentParser('Tensorboard to JSON')
    arg_parser.add_argument('--input',
                            type=str,
                            help='The path to the tensorboard folder (i.e. logdir)',
                            required=True)
    arg_parser.add_argument('--output',
                            type=str,
                            help='The path to the output folder',
                            default='.')  # default to the current folder

    arg_parser.add_argument("--prepare",
                            action="store_true",
                            default=False,
                            help="Prepare the output folder by creating it if it doesn't exist")

    args = arg_parser.parse_args()
    in_folder, out_folder = args.input, args.output

    if not os.path.exists(in_folder):
        print(f"Input folder {in_folder} not found")
        exit(-1)

    if not os.path.exists(out_folder):
        print("Creating output folder...")
        os.makedirs(out_folder, exist_ok=True)

    if args.prepare:
        in_folder = prepare_folders(in_folder)

    try:
        outfile = main(in_folder, out_folder)
    except Exception as e:
        print(f"Error: {e}")
        exit(-1)
    else:
        print(f"All data saved to {outfile}")
