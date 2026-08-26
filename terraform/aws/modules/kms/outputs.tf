output "key_id" {
  description = "KMS key ID."
  value       = aws_kms_key.platform.key_id
}

output "key_arn" {
  description = "KMS key ARN."
  value       = aws_kms_key.platform.arn
}
