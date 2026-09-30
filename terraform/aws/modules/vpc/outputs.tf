output "vpc_id" {
  value = aws_vpc.this.id
}

output "public_subnet_ids" {
  value = aws_subnet.public[*].id
}

output "private_db_subnet_ids" {
  description = "Private subnet IDs reserved for the RDS database."
  value       = aws_subnet.private_db[*].id
}
