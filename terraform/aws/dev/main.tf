module "ecr" {
  source = "../modules/ecr"

  project_name = var.project_name
  environment  = var.environment
}

module "s3" {
  source = "../modules/s3"

  project_name = var.project_name
  environment  = var.environment
  kms_key_arn  = module.kms.key_arn
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

  provider_credentials_secret_arn = module.secrets.provider_credentials_secret_arn
  gateway_api_key_secret_arn      = module.secrets.gateway_api_key_secret_arn
  environment_parameter_arn       = module.secrets.environment_parameter_arn
  log_level_parameter_arn         = module.secrets.log_level_parameter_arn
  default_provider_parameter_arn  = module.secrets.default_provider_parameter_arn

  s3_bucket_arn = module.s3.bucket_arn
  kms_key_arn   = module.kms.key_arn
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
  desired_count = var.ecs_desired_count
  vpc_id            = module.vpc.vpc_id
  public_subnet_ids = module.vpc.public_subnet_ids

  alb_security_group_id = module.alb.alb_security_group_id
  target_group_arn      = module.alb.target_group_arn

  execution_role_arn = module.iam.ecs_execution_role_arn
  task_role_arn      = module.iam.ecs_task_role_arn

  container_image = "${module.ecr.repository_url}:${var.image_tag}"

  s3_bucket_name = module.s3.bucket_name

  provider_credentials_secret_arn = module.secrets.provider_credentials_secret_arn
  gateway_api_key_secret_arn      = module.secrets.gateway_api_key_secret_arn
  environment_parameter_arn       = module.secrets.environment_parameter_arn
  log_level_parameter_arn         = module.secrets.log_level_parameter_arn
  default_provider_parameter_arn  = module.secrets.default_provider_parameter_arn

  alb_listener_dependency = module.alb.http_listener
}

module "secrets" {
  source = "../modules/secrets"

  name_prefix = "${var.project_name}/${var.environment}"
  environment = var.environment

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

module "kms" {
  source = "../modules/kms"

  name_prefix = "${var.project_name}-${var.environment}"

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}
