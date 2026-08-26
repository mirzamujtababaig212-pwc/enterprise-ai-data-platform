output "ecr_repository_url" {
  description = "ECR repository URL."
  value       = module.ecr.repository_url
}

output "ecr_repository_name" {
  description = "ECR repository name."
  value       = module.ecr.repository_name
}

output "s3_bucket_name" {
  description = "Platform S3 bucket."
  value       = module.s3.bucket_name
}

output "s3_bucket_arn" {
  description = "Platform S3 bucket ARN."
  value       = module.s3.bucket_arn
}

output "kms_key_id" {
  description = "Platform KMS key ID."
  value       = module.kms.key_id
}

output "kms_key_arn" {
  description = "Platform KMS key ARN."
  value       = module.kms.key_arn
}

output "provider_credentials_secret_arn" {
  description = "LLM provider credentials secret ARN."
  value       = module.secrets.provider_credentials_secret_arn
}

output "environment_parameter_arn" {
  description = "Environment parameter ARN."
  value       = module.secrets.environment_parameter_arn
}

output "log_level_parameter_arn" {
  description = "Log-level parameter ARN."
  value       = module.secrets.log_level_parameter_arn
}

output "default_provider_parameter_arn" {
  description = "Default-provider parameter ARN."
  value       = module.secrets.default_provider_parameter_arn
}

output "alb_dns_name" {
  description = "Application Load Balancer DNS name."
  value       = module.alb.alb_dns_name
}

output "ecs_cluster_name" {
  description = "ECS cluster name."
  value       = module.ecs.cluster_name
}

output "ecs_service_name" {
  description = "ECS service name."
  value       = module.ecs.service_name
}

output "gateway_url" {
  description = "LLM Gateway URL."
  value       = "http://${module.alb.alb_dns_name}"
}
