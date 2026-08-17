output "bucket_name" {
  description = "Platform S3 bucket name."
  value       = aws_s3_bucket.platform.bucket
}

output "bucket_arn" {
  description = "Platform S3 bucket ARN."
  value       = aws_s3_bucket.platform.arn
}
