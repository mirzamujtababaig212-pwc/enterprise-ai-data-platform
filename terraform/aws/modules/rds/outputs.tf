output "db_instance_arn" {
  description = "ARN of the PostgreSQL RDS instance."
  value       = aws_db_instance.this.arn
}

output "db_instance_id" {
  description = "Identifier of the PostgreSQL RDS instance."
  value       = aws_db_instance.this.id
}

output "endpoint" {
  description = "PostgreSQL RDS endpoint."
  value       = aws_db_instance.this.address
}

output "port" {
  description = "PostgreSQL RDS port."
  value       = aws_db_instance.this.port
}

output "db_name" {
  description = "Application database name."
  value       = aws_db_instance.this.db_name
}

output "master_username" {
  description = "RDS master username."
  value       = aws_db_instance.this.username
}

output "master_user_secret_arn" {
  description = "Secrets Manager ARN for the RDS-managed master password."
  value       = try(aws_db_instance.this.master_user_secret[0].secret_arn, null)
}

output "security_group_id" {
  description = "Security group ID for the RDS instance."
  value       = aws_security_group.rds.id
}

output "subnet_group_name" {
  description = "RDS DB subnet group name."
  value       = aws_db_subnet_group.this.name
}
