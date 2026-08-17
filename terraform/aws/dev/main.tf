module "ecr" {
  source = "../modules/ecr"

  project_name = var.project_name
  environment  = var.environment
}

module "s3" {
  source = "../modules/s3"

  project_name = var.project_name
  environment  = var.environment
}

output "ecr_repository_url" {
  description = "ECR repository URL."
  value       = module.ecr.repository_url
}

output "ecr_repository_name" {
  description = "ECR repository name."
  value       = module.ecr.repository_name
}

output "s3_bucket_name" {
  description = "Platform S3 bucket."
  value       = module.s3.bucket_name
}

output "s3_bucket_arn" {
  description = "Platform S3 bucket ARN."
  value       = module.s3.bucket_arn
}

module "vpc" {
  source = "../modules/vpc"

  project_name = var.project_name
  environment  = var.environment
  vpc_cidr     = "10.20.0.0/16"
}

module "iam" {
  source = "../modules/iam"

  project_name = var.project_name
  environment  = var.environment
}

module "alb" {
  source = "../modules/alb"

  project_name = var.project_name
  environment  = var.environment

  vpc_id            = module.vpc.vpc_id
  public_subnet_ids = module.vpc.public_subnet_ids
}

module "ecs" {
  source = "../modules/ecs"

  project_name = var.project_name
  environment  = var.environment
  aws_region   = var.aws_region

  vpc_id            = module.vpc.vpc_id
  public_subnet_ids = module.vpc.public_subnet_ids

  alb_security_group_id = module.alb.alb_security_group_id
  target_group_arn      = module.alb.target_group_arn

  execution_role_arn = module.iam.ecs_execution_role_arn
  task_role_arn      = module.iam.ecs_task_role_arn

  container_image = "${module.ecr.repository_url}:latest"

  s3_bucket_name = module.s3.bucket_name

  alb_listener_dependency = module.alb.http_listener
}
