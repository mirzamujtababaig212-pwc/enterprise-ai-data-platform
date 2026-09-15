output "cluster_name" {
  description = "ECS cluster name."
  value       = aws_ecs_cluster.this.name
}

output "service_name" {
  description = "ECS service name."
  value       = aws_ecs_service.gateway.name
}

output "task_definition_arn" {
  description = "ECS gateway task definition ARN."
  value       = aws_ecs_task_definition.gateway.arn
}

output "ecs_security_group_id" {
  description = "Security group ID for ECS tasks."
  value       = aws_security_group.ecs.id
}

output "batch_task_definition_arn" {
  description = "ARN of the ECS standalone batch task definition."
  value       = aws_ecs_task_definition.batch.arn
}

output "cluster_arn" {
  description = "ARN of the ECS cluster."
  value       = aws_ecs_cluster.this.arn
}
