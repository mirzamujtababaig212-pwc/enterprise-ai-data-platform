resource "aws_security_group" "rds" {
  name        = "${var.project_name}-${var.environment}-rds-sg"
  description = "Security group for the Enterprise AI Platform PostgreSQL database"
  vpc_id      = var.vpc_id

  ingress {
    description     = "PostgreSQL from ECS tasks"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [var.database_client_security_group_id]
  }

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

resource "aws_db_subnet_group" "this" {
  name       = "${var.project_name}-${var.environment}-db"
  subnet_ids = var.private_db_subnet_ids

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_db_instance" "this" {
  identifier = "${var.project_name}-${var.environment}-postgres"

  engine         = "postgres"
  engine_version = var.engine_version

  db_name  = var.db_name
  username = var.master_username

  manage_master_user_password = true

  instance_class        = var.instance_class
  allocated_storage     = var.allocated_storage
  max_allocated_storage = var.max_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = true

  db_subnet_group_name   = aws_db_subnet_group.this.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = false

  backup_retention_period = var.backup_retention_period
  deletion_protection     = var.deletion_protection
  skip_final_snapshot     = var.skip_final_snapshot

  apply_immediately = var.apply_immediately

  auto_minor_version_upgrade = true

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}
