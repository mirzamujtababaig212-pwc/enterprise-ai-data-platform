output "model_name" {
  description = "SageMaker model name."
  value       = aws_sagemaker_model.vehicle_risk.name
}

output "model_arn" {
  description = "SageMaker model ARN."
  value       = aws_sagemaker_model.vehicle_risk.arn
}

output "endpoint_name" {
  description = "SageMaker Vehicle Risk endpoint name."
  value       = aws_sagemaker_endpoint.vehicle_risk.name
}

output "endpoint_arn" {
  description = "SageMaker Vehicle Risk endpoint ARN."
  value       = aws_sagemaker_endpoint.vehicle_risk.arn
}
