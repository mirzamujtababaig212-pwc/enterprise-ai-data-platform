variable "cloud_sql_password" {
  description = "Write-only password for the Deldai Cloud SQL application user."
  type        = string
  sensitive   = true
}
