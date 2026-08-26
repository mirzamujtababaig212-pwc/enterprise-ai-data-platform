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

variable "image_tag" {
  description = "Immutable container image tag to deploy."
  type        = string
}

variable "ecs_desired_count" {
  description = "Number of ECS gateway tasks to run."
  type        = number
  default     = 1
}
