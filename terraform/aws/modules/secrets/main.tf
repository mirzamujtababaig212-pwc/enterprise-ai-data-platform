resource "aws_secretsmanager_secret" "provider_credentials" {
  name                    = "${var.name_prefix}/provider-credentials"
  description             = "Enterprise AI Platform LLM provider credentials"
  recovery_window_in_days = 7

  tags = var.tags
}

resource "aws_secretsmanager_secret" "gateway_api_key" {
  name                    = "${var.name_prefix}/gateway-api-key"
  description             = "Enterprise AI Platform Gateway API key"
  recovery_window_in_days = 7

  tags = var.tags
}

resource "aws_ssm_parameter" "environment" {
  name  = "/${var.name_prefix}/environment"
  type  = "String"
  value = var.environment

  tags = var.tags
}

resource "aws_ssm_parameter" "log_level" {
  name  = "/${var.name_prefix}/log-level"
  type  = "String"
  value = var.log_level

  tags = var.tags
}

resource "aws_ssm_parameter" "default_provider" {
  name  = "/${var.name_prefix}/default-provider"
  type  = "String"
  value = var.default_provider

  tags = var.tags
}
