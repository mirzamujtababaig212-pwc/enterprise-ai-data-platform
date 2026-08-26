variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "aws_region" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "public_subnet_ids" {
  type = list(string)
}

variable "alb_security_group_id" {
  type = string
}

variable "target_group_arn" {
  type = string
}

variable "execution_role_arn" {
  type = string
}

variable "task_role_arn" {
  type = string
}

variable "container_image" {
  type = string
}

variable "s3_bucket_name" {
  type = string
}

variable "cpu" {
  type    = number
  default = 1024
}

variable "memory" {
  type    = number
  default = 2048
}

variable "alb_listener_dependency" {
  type = any
}

variable "provider_credentials_secret_arn" {
  description = "ARN of the provider credentials secret."
  type        = string
}

variable "environment_parameter_arn" {
  description = "ARN of the environment SSM parameter."
  type        = string
}

variable "log_level_parameter_arn" {
  description = "ARN of the log-level SSM parameter."
  type        = string
}

variable "default_provider_parameter_arn" {
  description = "ARN of the default-provider SSM parameter."
  type        = string
}

variable "gateway_api_key_secret_arn" {
  description = "ARN of the gateway API key secret."
  type        = string
}

variable "desired_count" {
  description = "Number of ECS service tasks."
  type        = number
  default     = 1
}
