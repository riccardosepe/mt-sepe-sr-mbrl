import sys
import os


def main(args):
    if len(args) != 1:
        print("Usage: python make_export.py <path>")
        exit(-1)
    # first arg is a path
    path = args[0]

    # get project path
    # given path, it is the parent folder of the parent folder
    # while the current folder is not 'log'
    project_path = path
    env_name = os.path.basename(os.path.dirname(path))
    print(f"Exporting {env_name}")
    while os.path.basename(project_path) != "log":
        project_path = os.path.dirname(project_path)
        if project_path == "/":
            raise Exception("Could not find 'log' directory")

    # get all the seed folders
    seeds = os.listdir(path)
    # for each seed folder
    for seed in seeds:
        # get the path to the tensorboard folder
        tensorboard_path = os.path.join(path, seed, "tensorboard")
        # get the path to the model folder
        model_path = os.path.join(path, seed, "models")
        # get the path to the export folder
        export_path = os.path.join(project_path, "export", env_name, seed)
        # get the highest number of the .ckpt files
        # if model_path is empty, continue
        if len(os.listdir(model_path)) == 0:
            continue
        # create the export folder
        os.makedirs(export_path, exist_ok=True)
        max_ckpt = max([int(ckpt.split(".")[0]) for ckpt in os.listdir(model_path) if ckpt.split(".")[0] != 'emergency'])
        # get the path to the highest .ckpt file
        max_ckpt_path = os.path.join(model_path, f"{max_ckpt}.ckpt")
        # copy the tensorboard folder to the export folder
        os.system(f"cp -r {tensorboard_path} {export_path}")
        # copy the highest .ckpt file to the export folder
        os.system(f"cp {max_ckpt_path} {export_path}")

    print("Done")


if __name__ == '__main__':
    main(sys.argv[1:])
