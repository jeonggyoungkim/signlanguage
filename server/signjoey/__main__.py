import argparse
import os

import sys
from signjoey.training import train
from signjoey.prediction import test
import traceback

sys.path.append("/vol/research/extol/personal/cihan/code/SignJoey")


def main():
    try:
        ap = argparse.ArgumentParser("Joey NMT")

        ap.add_argument("mode", choices=["train", "test"], help="모델 학습 또는 테스트")

        ap.add_argument("config_path", type=str, help="YAML 설정 파일 경로")

        ap.add_argument("--ckpt", type=str, help="예측을 위한 체크포인트")

        ap.add_argument(
            "--output_path", type=str, help="번역 출력을 저장할 경로"
        )
        ap.add_argument("--gpu_id", type=str, default="0", help="작업을 실행할 GPU")
        args = ap.parse_args()

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

        if args.mode == "train":
            train(cfg_file=args.config_path)
        elif args.mode == "test":
            test(cfg_file=args.config_path, ckpt=args.ckpt, output_path=args.output_path)
        else:
            raise ValueError("알 수 없는 모드")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
