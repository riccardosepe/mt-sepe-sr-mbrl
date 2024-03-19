import os

from tqdm import trange
import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = "/Users/riccardo/PycharmProjects/TUDelft/branches/Physics_Informed_Model_Based_RL/FINAL/model_mlp"

    for i in trange(5):
        for lr in ["3e-03", "3e-04", "3e-05"]:
            base_dir = os.path.join(base_path, f"seed_{i}_lr_{lr}")
            file = os.path.join(base_dir, "emergency.ckpt")
            checkpoint = torch.load(file, map_location=device)
            smaller_ckpt = {}
            smaller_ckpt['transition_model'] = checkpoint['transition_model']
            smaller_ckpt['reward_model'] = checkpoint['reward_model']
            torch.save(smaller_ckpt, os.path.join(base_dir, "only_model.ckpt"))


if __name__ == '__main__':
    main()
