output "glue_role_arn" {
  description = "ARN of the dedicated AWS Glue service role."
  value       = aws_iam_role.glue.arn
}

output "catalog_database_name" {
  description = "AWS Glue Data Catalog database name."
  value       = aws_glue_catalog_database.enterprise.name
}
