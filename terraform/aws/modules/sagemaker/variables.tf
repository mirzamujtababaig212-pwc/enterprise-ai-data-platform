variable "project_name" {
  description = "Project name."
  type        = string
}

variable "environment" {
  description = "Deployment environment."
  type        = string
}

variable "container_image" {
  description = "Immutable SageMaker inference container image URI."
  type        = string
}

variable "model_artifact_s3_uri" {
  description = "S3 URI of the immutable SageMaker model artifact."
  type        = string
}

variable "execution_role_arn" {
  description = "SageMaker execution role ARN."
  type        = string
}

variable "model_name" {
  description = "SageMaker model resource name."
  type        = string
  default     = "vehicle-risk"
}

variable "endpoint_name" {
  description = "SageMaker endpoint name."
  type        = string
  default     = "vehicle-risk"
}

variable "serverless_memory_size_mb" {
  description = "Memory allocated to the SageMaker Serverless endpoint."
  type        = number
  default     = 1024
}

variable "serverless_max_concurrency" {
  description = "Maximum concurrent invocations for the SageMaker Serverless endpoint."
  type        = number
  default     = 1
}
