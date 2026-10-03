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

variable "artifact_bucket_name" {
  description = "Name of the S3 bucket containing Glue deployment artifacts."
  type        = string
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

variable "release_version" {
  description = "Immutable Glue application release version."
  type        = string
}

variable "wheel_name" {
  description = "Python wheel filename contained in the Glue release."
  type        = string
  default     = "enterprise_ai_platform-0.1.0-py3-none-any.whl"
}

variable "kms_key_arn" {
  description = "KMS key ARN used to encrypt the Glue artifact bucket."
  type        = string
}
