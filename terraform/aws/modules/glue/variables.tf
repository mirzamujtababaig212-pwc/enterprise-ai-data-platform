variable "project_name" {
  description = "Project name."
  type        = string
}

variable "environment" {
  description = "Deployment environment."
  type        = string
}

variable "s3_bucket_arn" {
  description = "ARN of the canonical enterprise data S3 bucket."
  type        = string
}

variable "catalog_database_name" {
  description = "AWS Glue Data Catalog database name."
  type        = string
  default     = "enterprise_ai_platform"
}

variable "artifact_bucket_arn" {
  description = "ARN of the S3 bucket containing Glue deployment artifacts."
  type        = string
}

variable "artifact_prefix" {
  description = "S3 key prefix containing immutable Glue deployment artifacts."
  type        = string
  default     = "glue/enterprise-ai-platform/"
}

variable "kms_key_arn" {
  description = "KMS key ARN used to encrypt the Glue artifact bucket."
  type        = string
}
