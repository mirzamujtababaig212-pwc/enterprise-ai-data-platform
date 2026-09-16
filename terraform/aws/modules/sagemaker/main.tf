resource "aws_sagemaker_model" "vehicle_risk" {
  name               = "${var.project_name}-${var.environment}-${var.model_name}"
  execution_role_arn = var.execution_role_arn

  primary_container {
    image          = var.container_image
    model_data_url = var.model_artifact_s3_uri

    environment = {
      MODEL_DIR = "/opt/ml/model"
      PORT      = "8080"
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
    Model       = var.model_name
  }
}

resource "aws_sagemaker_endpoint_configuration" "vehicle_risk" {
  name = "${var.project_name}-${var.environment}-${var.endpoint_name}"

  production_variants {
    variant_name = "AllTraffic"
    model_name   = aws_sagemaker_model.vehicle_risk.name

    serverless_config {
      memory_size_in_mb = var.serverless_memory_size_mb
      max_concurrency   = var.serverless_max_concurrency
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
    Endpoint    = var.endpoint_name
    Model       = var.model_name
  }
}

resource "aws_sagemaker_endpoint" "vehicle_risk" {
  name                 = "${var.project_name}-${var.environment}-${var.endpoint_name}"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.vehicle_risk.name

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
    Endpoint    = var.endpoint_name
    Model       = var.model_name
  }
}
