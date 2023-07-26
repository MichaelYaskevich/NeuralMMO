import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--id",
        type=str,
        default="96d8b5807255",
    )
    args = parser.parse_args()
    os.system(f'docker cp {args.id}:ijcai2022-nmmo-baselines/results loaded_results')