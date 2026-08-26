resource "aws_kms_key" "platform" {
  description             = "KMS key for Enterprise AI Platform"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  tags = var.tags
}

resource "aws_kms_alias" "platform" {
  name          = "alias/${var.name_prefix}"
  target_key_id = aws_kms_key.platform.key_id
}
