import argparse
import os
import sys
import boto3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", required=True, help="Example: ml/textcls")
    ap.add_argument("--model_version", required=True, help="Example: v1.0.0")
    ap.add_argument("--local_root", default="artifacts", help="Local artifacts root")
    args = ap.parse_args()

    local_dir = os.path.join(args.local_root, "versioned", args.model_version)
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(f"Missing local version dir: {local_dir}")

    required = ["metadata.json", "saved_model.tar.gz", "preprocess.joblib", "label_map.json"]
    for r in required:
        p = os.path.join(local_dir, r)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    s3 = boto3.client("s3", region_name=args.region)

    base_key = f"{args.prefix.strip('/').rstrip('/')}/versioned/{args.model_version}/"

    def upload(file_name: str):
        src = os.path.join(local_dir, file_name)
        key = base_key + file_name
        s3.upload_file(src, args.bucket, key)
        print(f"UPLOADED: s3://{args.bucket}/{key}")

    for fn in required:
        upload(fn)

    print("S3_VERSION_PREFIX:", f"s3://{args.bucket}/{base_key}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e), file=sys.stderr)
        raise
