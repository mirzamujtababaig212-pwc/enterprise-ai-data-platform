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
