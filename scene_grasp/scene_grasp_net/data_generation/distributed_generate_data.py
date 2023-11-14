import os
import argparse
import math
from pathlib import Path
from subprocess import Popen, PIPE
from scene_grasp.scene_grasp_net.data_generation.generate_data_nocs import add_generate_data_args
from common.utils.shape_utils import init_logs_dir, init_logs_dir_from_path
from common.config.config_global_paths import PROJECT_DATA_ROOT


def main():
    parser = argparse.ArgumentParser()
    add_generate_data_args(parser)
    parser.add_argument("--data_save_dir", type=str, default=None)
    parser.add_argument("--type", type=str, default="camera_train")
    args = parser.parse_args()

    worker_per_gpu = 12
    GPUS = [0, 1, 2, 3]
    print("=============> Using GPUS ", GPUS)
    workers = len(GPUS) * worker_per_gpu

    if args.type == "camera_train":
        list_all = (
            open(os.path.join(args.data_dir, "CAMERA", "train_list_all.txt"))
            .read()
            .splitlines()
        )
    elif args.type == "camera_val":
        list_all = (
            open(os.path.join(args.data_dir, "CAMERA", "val_list_all.txt"))
            .read()
            .splitlines()
        )
        list_all = list_all[:1000]
    elif args.type == "real_train":
        list_all = (
            open(os.path.join(args.data_dir, "Real", "train_list_all.txt"))
            .read()
            .splitlines()
        )
    else:
        list_all = (
            open(os.path.join(args.data_dir, "Real", "test_list_all.txt"))
            .read()
            .splitlines()
        )

    logger_name = "gen_centersnap_data"
    if args.data_save_dir is None:
        data_save_root = PROJECT_DATA_ROOT / "scenegraspnet_preprocessed_data"
        model_path = Path(args.model_path)
        suffix = f"{model_path.parent.name}__{model_path.stem}"
        data_save_dir, logger = init_logs_dir(
            data_save_root, logger_name, suffix, vars(args)
        )
    else:
        data_save_dir = Path(args.data_save_dir)
        data_save_dir, logger = init_logs_dir_from_path(
            data_save_dir, logger_name, vars(args)
        )

    all_frames = range(0, len(list_all))
    frames_per_worker = math.ceil(len(all_frames) / workers)
    for i in range(workers):
        curr_gpu = GPUS[i // worker_per_gpu]

        start = i * frames_per_worker
        end = start + frames_per_worker

        print(i, curr_gpu)
        print(all_frames[start:end])
        print("start, : end", start, end)

        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = str(curr_gpu)
        command = [
            "python",
            "scene_grasp_net/data_generation/distributed_worker.py",
            "--data_dir",
            str(args.data_dir),
            "--model_path",
            str(args.model_path),
            "--object_deformnet_nocs_results_dir",
            str(args.object_deformnet_nocs_results_dir),
            "--data_save_dir",
            str(data_save_dir.absolute()),
            "--type",
            str(args.type),
            "--id",
            str(i),
            "--start",
            str(start),
            "--end",
            str(end),
            "--all_frames",
            str(len(list_all)),
        ]
        if args.gen_small_sample:
            command.append("--gen_small_sample")
        print(command)

        log = open(data_save_dir / f"{args.type}_worker_{i}.txt", "w")
        Popen(command, env=my_env, stderr=log, stdout=log)


if __name__ == "__main__":
    main()
