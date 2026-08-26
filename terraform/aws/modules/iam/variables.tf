variable "project_name" {
  description = "Project name."
  type        = string
}

variable "environment" {
  description = "Deployment environment."
  type        = string
}

variable "s3_bucket_arn" {
  description = "ARN of the platform S3 bucket."
  type        = string
}

variable "provider_credentials_secret_arn" {
  description = "ARN of the Secrets Manager secret containing LLM provider credentials."
  type        = string
}

variable "environment_parameter_arn" {
  description = "ARN of the SSM environment parameter."
  type        = string
}

variable "log_level_parameter_arn" {
  description = "ARN of the SSM log-level parameter."
  type        = string
}

variable "default_provider_parameter_arn" {
  description = "ARN of the SSM default-provider parameter."
  type        = string
}

variable "kms_key_arn" {
  description = "ARN of the platform KMS key."
  type        = string
}

variable "gateway_api_key_secret_arn" {
  description = "ARN of the gateway API key secret."
  type        = string
}
