import sys
import os


def main(args):
    if len(args) != 1:
        print("Usage: python make_export.py <path>")
        exit(-1)
    # first arg is a path
    path = args[0]
    # path has the following structure:
    # path
    #   - seed_x
    #       - model
    #           - n1.ckpt
    #           - n2.ckpt
    #           - ...
    #       - tensorboard
    #           - tensorboard outfile

    # for each seed folder, get both the tensorboard folder and the .ckpt file with the highest number and put
    # them in the same folder, for each seed, in a new folder called "export" with the following structure:
    # export
    #   - seed_x
    #       - tensorboard
    #           - tensorboard outfile
    #       - nmax.ckpt
    #   - seed_y
    #       - tensorboard
    #           - tensorboard outfile
    #       - nmax.ckpt
    #   - ...

    # get all the seed folders
    seeds = os.listdir(path)
    # for each seed folder
    for seed in seeds:
        # get the path to the tensorboard folder
        tensorboard_path = os.path.join(path, seed, "tensorboard")
        # get the path to the model folder
        model_path = os.path.join(path, seed, "model")
        # get the path to the export folder
        export_path = os.path.join(path, "export", seed)
        # create the export folder
        os.makedirs(export_path, exist_ok=True)
        # get the highest number of the .ckpt files
        max_ckpt = max([int(ckpt.split(".")[0]) for ckpt in os.listdir(model_path)])
        # get the path to the highest .ckpt file
        max_ckpt_path = os.path.join(model_path, f"{max_ckpt}.ckpt")
        # copy the tensorboard folder to the export folder
        os.system(f"cp -r {tensorboard_path} {export_path}")
        # copy the highest .ckpt file to the export folder
        os.system(f"cp {max_ckpt_path} {export_path}")


if __name__ == '__main__':
    main(sys.argv[1:])
