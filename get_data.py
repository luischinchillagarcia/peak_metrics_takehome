from dataclasses import dataclass
from dotenv import load_dotenv
import os
import boto3


load_dotenv()


@dataclass
class Env:
    BUCKET_NAME = os.getenv('BUCKET_NAME', '')
    S3_PATH = os.getenv('S3_PATH', '')
    ACCESS_KEY = os.getenv('ACCESS_KEY', '')
    ACCESS_SECRET = os.getenv('ACCESS_SECRET', '')


def get_s3_bucket(key: str, secret: str, bucket_name: str):
    session = boto3.Session(
        aws_access_key_id=key,
        aws_secret_access_key=secret,
    )
    s3 = session.resource('s3')
    bucket = s3.Bucket(bucket_name) #type: ignore
    
    return bucket


def download_s3_path(bucket, s3_path):
    for obj in bucket.objects.filter(Prefix=s3_path):
        directory_name = os.path.dirname(obj.key)
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        bucket.download_file(obj.key, obj.key) 
        
    return None


def main():
    bucket = get_s3_bucket(key=Env.ACCESS_KEY, secret=Env.ACCESS_SECRET, bucket_name=Env.BUCKET_NAME)
    download_s3_path(bucket=bucket, s3_path=Env.S3_PATH)
    

if __name__ == '__main__':
    main()
