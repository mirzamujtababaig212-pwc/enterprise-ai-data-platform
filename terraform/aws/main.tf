terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  description = "AWS deployment region."
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment."
  type        = string
  default     = "dev"
}

variable "bucket_name" {
  description = "Globally unique platform S3 bucket."
  type        = string
  default     = "enterprise-data-ai-platform"
}

module "s3" {
  source = "../s3"

  bucket_name = var.bucket_name
  environment = var.environment
  region      = var.aws_region
}

output "s3_bucket" {
  description = "Enterprise AI Platform S3 bucket name"
  value       = module.s3.bucket_name
}

output "s3_bucket_arn" {
  description = "Enterprise AI Platform S3 bucket ARN"
  value       = module.s3.bucket_arn
}

output "s3_bucket_region" {
  description = "Enterprise AI Platform S3 bucket region"
  value       = module.s3.bucket_region
}
