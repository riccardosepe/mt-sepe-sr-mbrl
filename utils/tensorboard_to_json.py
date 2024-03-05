"""
Save tensorboard scalars to a JSON file.
"""
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from argparse import ArgumentParser
import json


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

    args = arg_parser.parse_args()
    in_folder, out_folder = args.input, args.output

    if not os.path.exists(in_folder):
        print(f"Input folder {in_folder} not found")
        exit(-1)

    if not os.path.exists(out_folder):
        print("Creating output folder...")
        os.makedirs(out_folder, exist_ok=True)

    try:
        outfile = main(in_folder, out_folder)
    except Exception as e:
        print(f"Error: {e}")
        exit(-1)
    else:
        print(f"All data saved to {outfile}")
