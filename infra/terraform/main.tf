terraform {
  required_version = ">= 1.6.0"
}

variable "environment" {
  type = string
}

variable "image_tag" {
  type = string
}

resource "null_resource" "network" {
  triggers = { env = var.environment }
}

resource "null_resource" "database" {
  triggers = { env = var.environment, engine = "postgresql-pgvector" }
}

resource "null_resource" "api_service" {
  triggers = { image = var.image_tag, env = var.environment }
}

resource "null_resource" "monitoring" {
  triggers = { env = var.environment }
}

output "environment" {
  value = var.environment
}
