import torch
import numpy
import argparse
from scene_grasp.scene_grasp_net.data_generation.generate_data_nocs import (
    annotate_camera_train,
    annotate_test_data,
    annotate_real_train,
    add_generate_data_args,
)

seed = 1
numpy.random.seed(seed)
torch.manual_seed(seed)


def main(args):
    if args.end > args.all_frames:
        args.end = args.all_frames
    if args.type == "camera_train":
        annotate_camera_train(
            args.data_dir,
            args.data_save_dir,
            args.model_path,
            args.start,
            args.end,
            args.gen_small_sample,
        )
    elif args.type == "camera_val":
        annotate_test_data(
            args.data_dir,
            args.data_save_dir,
            "CAMERA",
            "val",
            args.model_path,
            args.start,
            args.end,
            args.object_deformnet_nocs_results_dir,
            args.gen_small_sample,
        )
    elif args.type == "real_train":
        annotate_real_train(
            args.data_dir,
            args.data_save_dir,
            args.model_path,
            args.start,
            args.end,
            args.gen_small_sample
        )
    else:
        annotate_test_data(
            args.data_dir,
            args.data_save_dir,
            "Real",
            "test",
            args.model_path,
            args.start,
            args.end,
            args.object_deformnet_nocs_results_dir,
            args.gen_small_sample,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generate_data_args(parser)
    parser.add_argument("--data_save_dir", type=str, help="where data is being saved")
    parser.add_argument("--type", type=str, default="train")
    parser.add_argument("--id", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=0, type=int)
    parser.add_argument("--all_frames", default=0, type=int)
    args = parser.parse_args()
    main(args)
