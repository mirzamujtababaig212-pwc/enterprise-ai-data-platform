module "kms" {
  source = "../modules/kms"

  name_prefix = "${var.project_name}-${var.environment}"

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

module "s3" {
  source = "../modules/s3"

  project_name = var.project_name
  environment  = var.environment
  kms_key_arn  = module.kms.key_arn
}

data "aws_s3_bucket" "enterprise_data" {
  bucket = "enterprise-data-ai-platform"
}

module "glue" {
  source = "../modules/glue"

  project_name          = var.project_name
  environment           = var.environment
  s3_bucket_arn         = data.aws_s3_bucket.enterprise_data.arn
  artifact_bucket_arn   = module.s3.bucket_arn
  artifact_prefix       = "glue/enterprise-ai-platform/"
  kms_key_arn           = module.kms.key_arn
  catalog_database_name = "enterprise_ai_platform"
}

module "ecr" {
  source = "../modules/ecr"

  project_name = var.project_name
  environment  = var.environment
}

module "vpc" {
  source = "../modules/vpc"

  project_name = var.project_name
  environment  = var.environment
  vpc_cidr     = "10.20.0.0/16"
}

resource "aws_security_group" "database_client" {
  name        = "${var.project_name}-${var.environment}-database-client-sg"
  description = "Security group for workloads that connect to the Enterprise AI Platform PostgreSQL database"
  vpc_id      = module.vpc.vpc_id

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

module "iam" {
  source = "../modules/iam"

  project_name = var.project_name
  environment  = var.environment

  bedrock_invoke_resource_arns = [
    "arn:aws:bedrock:${var.aws_region}:715621342004:inference-profile/us.amazon.nova-micro-v1:0",
    "arn:aws:bedrock:${var.aws_region}::foundation-model/amazon.nova-micro-v1:0",
    "arn:aws:bedrock:us-east-2::foundation-model/amazon.nova-micro-v1:0",
    "arn:aws:bedrock:us-west-2::foundation-model/amazon.nova-micro-v1:0",
    "arn:aws:bedrock:${var.aws_region}::foundation-model/amazon.titan-embed-text-v2:0",
  ]

  provider_credentials_secret_arn = module.secrets.provider_credentials_secret_arn
  gateway_api_key_secret_arn      = module.secrets.gateway_api_key_secret_arn
  rds_master_user_secret_arn      = module.rds.master_user_secret_arn
  environment_parameter_arn       = module.secrets.environment_parameter_arn
  log_level_parameter_arn         = module.secrets.log_level_parameter_arn
  default_provider_parameter_arn  = module.secrets.default_provider_parameter_arn

  s3_bucket_arn = module.s3.bucket_arn
  kms_key_arn   = module.kms.key_arn

  ecr_repository_arn        = module.ecr.repository_arn
  model_artifact_bucket_arn = module.s3.bucket_arn
  model_artifact_prefix     = "model-artifacts/"

  data_bucket_arn = data.aws_s3_bucket.enterprise_data.arn
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

  project_name      = var.project_name
  environment       = var.environment
  aws_region        = var.aws_region
  desired_count     = var.ecs_desired_count
  vpc_id            = module.vpc.vpc_id
  public_subnet_ids = module.vpc.public_subnet_ids

  alb_security_group_id = module.alb.alb_security_group_id
  target_group_arn      = module.alb.target_group_arn

  additional_security_group_ids = [
    aws_security_group.database_client.id
  ]

  execution_role_arn = module.iam.ecs_execution_role_arn
  task_role_arn      = module.iam.ecs_task_role_arn

  container_image = "${module.ecr.repository_url}:${var.image_tag}"

  s3_bucket_name = module.s3.bucket_name

  provider_credentials_secret_arn = module.secrets.provider_credentials_secret_arn
  gateway_api_key_secret_arn      = module.secrets.gateway_api_key_secret_arn
  postgres_db                     = module.rds.db_name
  postgres_user                   = module.rds.master_username
  postgres_host                   = module.rds.endpoint
  postgres_port                   = module.rds.port
  postgres_password_secret_arn    = module.rds.master_user_secret_arn
  environment_parameter_arn       = module.secrets.environment_parameter_arn
  log_level_parameter_arn         = module.secrets.log_level_parameter_arn
  default_provider_parameter_arn  = module.secrets.default_provider_parameter_arn

  alb_listener_dependency = module.alb.http_listener

  batch_execution_role_arn = module.iam.ecs_batch_execution_role_arn
  batch_task_role_arn      = module.iam.ecs_batch_task_role_arn
  batch_container_image    = "${module.ecr.repository_url}:aws-batch-merge-20260915-095445"
  batch_data_bucket_name   = data.aws_s3_bucket.enterprise_data.bucket
}

module "rds" {
  source = "../modules/rds"

  project_name = var.project_name
  environment  = var.environment

  vpc_id                            = module.vpc.vpc_id
  private_db_subnet_ids             = module.vpc.private_db_subnet_ids
  database_client_security_group_id = aws_security_group.database_client.id
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

module "step_functions" {
  source = "../modules/step_functions"

  project_name = var.project_name
  environment  = var.environment
  aws_region   = var.aws_region

  cluster_arn               = module.ecs.cluster_arn
  batch_task_definition_arn = module.ecs.batch_task_definition_arn

  batch_execution_role_arn = module.iam.ecs_batch_execution_role_arn
  batch_task_role_arn      = module.iam.ecs_batch_task_role_arn

  subnet_ids        = module.vpc.public_subnet_ids
  security_group_id = module.ecs.ecs_security_group_id
}

module "sagemaker" {
  source = "../modules/sagemaker"

  project_name = var.project_name
  environment  = var.environment

  container_image = "${module.ecr.repository_url}:sagemaker-vehicle-risk-model-v2-docker"

  model_artifact_s3_uri = var.sagemaker_model_artifact_s3_uri

  execution_role_arn = module.iam.sagemaker_execution_role_arn

  model_name = "vehicle-risk"
}

output "sagemaker_vehicle_risk_endpoint_name" {
  description = "SageMaker Vehicle Risk endpoint name."
  value       = module.sagemaker.endpoint_name
}

output "sagemaker_vehicle_risk_endpoint_arn" {
  description = "SageMaker Vehicle Risk endpoint ARN."
  value       = module.sagemaker.endpoint_arn
}
