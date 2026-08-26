output "provider_credentials_secret_arn" {
  description = "ARN of the LLM provider credentials secret."
  value       = aws_secretsmanager_secret.provider_credentials.arn
}

output "environment_parameter_arn" {
  description = "ARN of the environment SSM parameter."
  value       = aws_ssm_parameter.environment.arn
}

output "log_level_parameter_arn" {
  description = "ARN of the log-level SSM parameter."
  value       = aws_ssm_parameter.log_level.arn
}

output "default_provider_parameter_arn" {
  description = "ARN of the default-provider SSM parameter."
  value       = aws_ssm_parameter.default_provider.arn
}

output "gateway_api_key_secret_arn" {
  description = "ARN of the gateway API key secret."
  value       = aws_secretsmanager_secret.gateway_api_key.arn
}
