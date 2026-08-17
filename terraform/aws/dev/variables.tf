variable "aws_region" {
  description = "AWS region where the platform will be deployed."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name."
  type        = string
  default     = "enterprise-ai-platform"
}

variable "environment" {
  description = "Deployment environment."
  type        = string
  default     = "dev"
}
