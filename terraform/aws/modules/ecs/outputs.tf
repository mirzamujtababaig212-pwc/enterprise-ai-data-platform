output "cluster_name" {
  value = aws_ecs_cluster.this.name
}

output "service_name" {
  value = aws_ecs_service.gateway.name
}

output "task_definition_arn" {
  value = aws_ecs_task_definition.gateway.arn
}

output "ecs_security_group_id" {
  value = aws_security_group.ecs.id
}
