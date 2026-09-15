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
