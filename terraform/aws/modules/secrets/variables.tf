variable "name_prefix" {
  description = "Prefix used for AWS Secrets Manager and SSM resources."
  type        = string
}

variable "environment" {
  description = "Deployment environment."
  type        = string
}

variable "tags" {
  description = "Tags applied to resources."
  type        = map(string)
  default     = {}
}

variable "log_level" {
  description = "Application log level."
  type        = string
  default     = "INFO"
}

variable "default_provider" {
  description = "Default LLM provider."
  type        = string
  default     = "openai"
}
