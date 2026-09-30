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

variable "data_bucket_arn" {
  description = "ARN of the canonical enterprise data S3 bucket used by ECS batch tasks."
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

variable "rds_master_user_secret_arn" {
  description = "ARN of the RDS-managed Secrets Manager secret containing the PostgreSQL master password."
  type        = string
}

variable "bedrock_model_arns" {
  description = "Bedrock foundation-model ARNs that the gateway ECS task may invoke."
  type        = list(string)
  default     = []
}

variable "bedrock_invoke_resource_arns" {
  description = "Bedrock foundation-model and inference-profile ARNs that the gateway ECS task may invoke."
  type        = list(string)
  default     = []
}

variable "ecr_repository_arn" {
  description = "ARN of the ECR repository containing SageMaker inference images."
  type        = string
}

variable "model_artifact_bucket_arn" {
  description = "ARN of the S3 bucket containing SageMaker model artifacts."
  type        = string
}

variable "model_artifact_prefix" {
  description = "S3 key prefix containing SageMaker model artifacts."
  type        = string
  default     = "model-artifacts/"
}
