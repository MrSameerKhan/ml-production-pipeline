import argparse
import time
import boto3


def wait_endpoint(sm, endpoint_name: str, timeout_sec: int = 1800):
    start = time.time()
    while True:
        desc = sm.describe_endpoint(EndpointName=endpoint_name)
        status = desc["EndpointStatus"]
        print("EndpointStatus:", status)
        if status in ("InService", "Failed"):
            return status, desc
        if time.time() - start > timeout_sec:
            raise TimeoutError("Endpoint deployment timeout")
        time.sleep(30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--exec-role-arn", required=True)
    ap.add_argument("--instance-type", required=True)
    ap.add_argument("--initial-count", type=int, required=True)

    ap.add_argument("--ecr-image", required=True, help="Full ECR image URI")
    ap.add_argument("--model-data-s3", required=True, help="s3://.../saved_model.tar.gz")

    ap.add_argument("--model-name", default=None)
    ap.add_argument("--config-name", default=None)
    args = ap.parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)

    # Names
    model_name = args.model_name or f"{args.endpoint}-model-{int(time.time())}"
    cfg_name = args.config_name or f"{args.endpoint}-cfg-{int(time.time())}"

    print("MODEL_NAME:", model_name)
    print("ENDPOINT_CONFIG:", cfg_name)

    # Create model
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": args.ecr_image,
            "ModelDataUrl": args.model_data_s3
        },
        ExecutionRoleArn=args.exec_role_arn,
    )
    print("Created model.")

    # Create endpoint config
    sm.create_endpoint_config(
        EndpointConfigName=cfg_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InitialInstanceCount": args.initial_count,
            "InstanceType": args.instance_type,
        }]
    )
    print("Created endpoint config.")

    # Create or update endpoint
    try:
        sm.describe_endpoint(EndpointName=args.endpoint)
        print("Updating endpoint:", args.endpoint)
        sm.update_endpoint(EndpointName=args.endpoint, EndpointConfigName=cfg_name)
    except sm.exceptions.ClientError:
        print("Creating endpoint:", args.endpoint)
        sm.create_endpoint(EndpointName=args.endpoint, EndpointConfigName=cfg_name)

    status, desc = wait_endpoint(sm, args.endpoint)
    if status != "InService":
        raise RuntimeError(f"Deployment failed: {desc}")

    print("DEPLOYED_ENDPOINT:", args.endpoint)


if __name__ == "__main__":
    main()
