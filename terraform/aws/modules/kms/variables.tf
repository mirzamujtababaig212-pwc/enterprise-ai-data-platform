variable "name_prefix" {
  description = "Prefix for KMS resources."
  type        = string
}

variable "tags" {
  description = "Tags applied to resources."
  type        = map(string)
  default     = {}
}
