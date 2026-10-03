variable "project_name" {
  description = "Project name."
  type        = string
}

variable "environment" {
  description = "Deployment environment."
  type        = string
}

variable "aws_region" {
  description = "AWS region."
  type        = string
}

variable "cluster_arn" {
  description = "ECS cluster ARN used by the state machine."
  type        = string
}

variable "batch_task_definition_arn" {
  description = "ECS batch task definition ARN."
  type        = string
}

variable "batch_execution_role_arn" {
  description = "ECS batch task execution role ARN."
  type        = string
}

variable "batch_task_role_arn" {
  description = "ECS batch task role ARN."
  type        = string
}

variable "subnet_ids" {
  description = "Public subnet IDs used by Fargate batch tasks."
  type        = list(string)
}

variable "security_group_id" {
  description = "Security group ID used by Fargate batch tasks."
  type        = string
}

variable "glue_job_name" {
  description = "AWS Glue job name used by the Glue execution path."
  type        = string
}

variable "glue_artifact_bucket_name" {
  description = "S3 bucket containing immutable Glue application releases."
  type        = string
}

variable "glue_release_version" {
  description = "Immutable Glue application release version used by Step Functions Glue runs."
  type        = string
}
