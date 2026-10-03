resource "aws_iam_role" "glue" {
  name = "${var.project_name}-${var.environment}-glue-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Effect = "Allow"

        Principal = {
          Service = "glue.amazonaws.com"
        }

        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

resource "aws_iam_role_policy" "glue_data_access" {
  name = "${var.project_name}-${var.environment}-glue-data-access"
  role = aws_iam_role.glue.id

  policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Sid    = "S3BucketAccess"
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = var.s3_bucket_arn
      },
      {
        Sid    = "S3ObjectAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = "${var.s3_bucket_arn}/*"
      },
      {
        Sid    = "GlueCatalogTableAccess"
        Effect = "Allow"
        Action = [
          "glue:GetTable",
          "glue:CreateTable",
          "glue:UpdateTable"
        ]
        Resource = "*"
      },
      {
        Sid    = "GlueArtifactRead"
        Effect = "Allow"
        Action = [
          "s3:GetObject"
        ]
        Resource = "${var.artifact_bucket_arn}/${var.artifact_prefix}*"
      },
      {
        Sid    = "GlueArtifactDecrypt"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = var.kms_key_arn
      }
    ]
  })
}

resource "aws_glue_catalog_database" "enterprise" {
  name        = var.catalog_database_name
  description = "DELDAI enterprise AWS Glue Data Catalog database."
}

resource "aws_glue_job" "pipeline" {
  name        = "${var.project_name}-${var.environment}-pipeline"
  description = "Enterprise AI Platform pipeline execution on AWS Glue."

  role_arn = aws_iam_role.glue.arn

  glue_version      = "5.1"
  worker_type       = "G.1X"
  number_of_workers = 2
  max_retries       = 1
  timeout           = 120
  execution_class   = "STANDARD"

  command {
    name            = "glueetl"
    script_location = "s3://${var.artifact_bucket_name}/${var.artifact_prefix}${var.release_version}/scripts/pipeline_job.py"
    python_version  = "3"
  }

  default_arguments = {
    "--PIPELINE_NAME" = "bronze"
    "--MODE"          = "batch"
    "--APP_ENV"       = "aws"

    "--ARTIFACT_BUCKET" = var.artifact_bucket_name
    "--RELEASE_VERSION" = var.release_version

    "--extra-py-files" = "s3://${var.artifact_bucket_name}/${var.artifact_prefix}${var.release_version}/wheels/${var.wheel_name}"

    "--enable-continuous-cloudwatch-log" = "true"
    "--enable-spark-ui"                  = "true"
  }

  execution_property {
    max_concurrent_runs = 1
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}
