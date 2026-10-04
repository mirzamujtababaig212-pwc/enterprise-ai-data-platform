resource "google_storage_bucket" "deldai" {
  name                        = "deldai-enterprise-ai-dev"
  location                    = "US-EAST1"
  project                     = "api-project-212575571422"
  uniform_bucket_level_access = true
  versioning {
    enabled = true
  }
}

resource "google_service_account" "platform" {
  account_id   = "deldai-platform"
  display_name = "Deldai Enterprise AI Platform"
  description  = "Deldai Enterprise AI OS platform service account"
  project      = "api-project-212575571422"
}

resource "google_artifact_registry_repository" "deldai" {
  location      = "us-east1"
  repository_id = "deldai"
  description   = "Deldai Enterprise AI OS container images"
  format        = "DOCKER"
  project       = "api-project-212575571422"
}

resource "google_storage_bucket_iam_member" "platform_object_admin" {
  bucket = google_storage_bucket.deldai.name
  role   = "roles/storage.objectAdmin"
  member = google_service_account.platform.member
}

resource "google_artifact_registry_repository_iam_member" "platform_writer" {
  project    = google_artifact_registry_repository.deldai.project
  location   = google_artifact_registry_repository.deldai.location
  repository = google_artifact_registry_repository.deldai.repository_id
  role       = "roles/artifactregistry.writer"
  member     = google_service_account.platform.member
}

resource "google_sql_database_instance" "control_plane" {
  name             = "deldai-enterprise-ai-dev"
  database_version = "POSTGRES_16"
  region           = "us-east1"
  project          = "api-project-212575571422"

  deletion_protection = true

  settings {
    tier              = "db-f1-micro"
    edition           = "ENTERPRISE"
    availability_type = "ZONAL"
    disk_type         = "PD_SSD"
    disk_size         = 10
    disk_autoresize   = true

    ip_configuration {
      ipv4_enabled    = false
      private_network = "projects/api-project-212575571422/global/networks/default"
    }
  }

  depends_on = [
    google_service_networking_connection.private_service_access
  ]
}


resource "google_compute_global_address" "private_service_access" {
  name          = "deldai-private-service-access"
  project       = "api-project-212575571422"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = "projects/api-project-212575571422/global/networks/default"
}

resource "google_service_networking_connection" "private_service_access" {
  network                 = "projects/api-project-212575571422/global/networks/default"
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_service_access.name]
}

resource "google_sql_database" "enterprise_ai" {
  name     = "enterprise_ai"
  instance = google_sql_database_instance.control_plane.name
  project  = "api-project-212575571422"
}

resource "google_sql_user" "enterprise_ai" {
  name     = "enterprise_ai"
  instance = google_sql_database_instance.control_plane.name
  project  = "api-project-212575571422"

  password_wo         = var.cloud_sql_password
  password_wo_version = 1
}

resource "google_secret_manager_secret_iam_member" "db_password_accessor" {
  project   = "api-project-212575571422"
  secret_id = "deldai-enterprise-ai-dev-db-password"
  role      = "roles/secretmanager.secretAccessor"
  member    = google_service_account.platform.member
}


resource "google_secret_manager_secret_iam_member" "provider_credentials_accessor" {
  project   = "api-project-212575571422"
  secret_id = "deldai-enterprise-ai-dev-provider-credentials"
  role      = "roles/secretmanager.secretAccessor"
  member    = google_service_account.platform.member
}

resource "google_secret_manager_secret_iam_member" "api_key_accessor" {
  project   = "api-project-212575571422"
  secret_id = "deldai-enterprise-ai-dev-api-key"
  role      = "roles/secretmanager.secretAccessor"
  member    = google_service_account.platform.member
}

resource "google_cloud_run_v2_service" "gateway" {
  name     = "deldai-enterprise-ai-dev-gateway"
  location = "us-east1"
  project  = "api-project-212575571422"

  deletion_protection = false

  template {
    service_account = google_service_account.platform.email

    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }

    containers {
      image = "us-east1-docker.pkg.dev/api-project-212575571422/deldai/enterprise-ai-platform@sha256:f8b1af1485937bc52e4acf968e38a2840f5ded4b485d0566566c833ee0bcae98"

      ports {
        container_port = 8000
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }

      env {
        name  = "ENABLED_PROVIDERS"
        value = "openai"
      }

      env {
        name  = "DEFAULT_PROVIDER"
        value = "openai"
      }

      env {
        name  = "ENVIRONMENT"
        value = "gcp-dev"
      }

      env {
        name = "API_KEY"
        value_source {
          secret_key_ref {
            secret  = "deldai-enterprise-ai-dev-api-key"
            version = "1"
          }
        }
      }

      env {
        name = "PROVIDER_CREDENTIALS"

        value_source {
          secret_key_ref {
            secret  = "deldai-enterprise-ai-dev-provider-credentials"
            version = "1"
          }
        }
      }

      env {
        name  = "POSTGRES_HOST"
        value = google_sql_database_instance.control_plane.private_ip_address
      }

      env {
        name  = "POSTGRES_PORT"
        value = "5432"
      }

      env {
        name  = "POSTGRES_DB"
        value = google_sql_database.enterprise_ai.name
      }

      env {
        name  = "POSTGRES_USER"
        value = google_sql_user.enterprise_ai.name
      }

      env {
        name = "POSTGRES_PASSWORD"

        value_source {
          secret_key_ref {
            secret  = "deldai-enterprise-ai-dev-db-password"
            version = "1"
          }
        }
      }

      startup_probe {
        initial_delay_seconds = 5
        timeout_seconds       = 5
        period_seconds        = 10
        failure_threshold     = 6

        http_get {
          path = "/healthz"
          port = 8000
        }
      }
    }

    vpc_access {
      egress = "PRIVATE_RANGES_ONLY"

      network_interfaces {
        network    = "default"
        subnetwork = "default"
      }
    }
  }

  depends_on = [
    google_secret_manager_secret_iam_member.provider_credentials_accessor,
    google_secret_manager_secret_iam_member.db_password_accessor,
    google_secret_manager_secret_iam_member.api_key_accessor,
  ]
}


resource "google_secret_manager_secret_iam_member" "control_plane_api_key_tenants_accessor" {
  project   = "api-project-212575571422"
  secret_id = "deldai-enterprise-ai-dev-control-plane-api-key-tenants"
  role      = "roles/secretmanager.secretAccessor"
  member    = google_service_account.platform.member
}

resource "google_cloud_run_v2_service" "control_plane" {
  name     = "deldai-enterprise-ai-dev-control-plane"
  location = "us-east1"
  project  = "api-project-212575571422"

  deletion_protection = false

  template {
    service_account = google_service_account.platform.email

    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }

    containers {
      image = "us-east1-docker.pkg.dev/api-project-212575571422/deldai/enterprise-ai-platform@sha256:f8b1af1485937bc52e4acf968e38a2840f5ded4b485d0566566c833ee0bcae98"

      command = ["uvicorn"]

      args = [
        "app.control_plane.app:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
      ]

      ports {
        container_port = 8000
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }

      env {
        name  = "ENABLED_PROVIDERS"
        value = "openai"
      }

      env {
        name  = "DEFAULT_PROVIDER"
        value = "openai"
      }

      env {
        name  = "ENVIRONMENT"
        value = "gcp-dev"
      }

      env {
        name = "API_KEY"

        value_source {
          secret_key_ref {
            secret  = "deldai-enterprise-ai-dev-api-key"
            version = "1"
          }
        }
      }

      env {
        name = "CONTROL_PLANE_API_KEY_TENANTS"

        value_source {
          secret_key_ref {
            secret  = "deldai-enterprise-ai-dev-control-plane-api-key-tenants"
            version = "1"
          }
        }
      }

      env {
        name = "PROVIDER_CREDENTIALS"

        value_source {
          secret_key_ref {
            secret  = "deldai-enterprise-ai-dev-provider-credentials"
            version = "1"
          }
        }
      }

      env {
        name  = "POSTGRES_HOST"
        value = google_sql_database_instance.control_plane.private_ip_address
      }

      env {
        name  = "POSTGRES_PORT"
        value = "5432"
      }

      env {
        name  = "POSTGRES_DB"
        value = google_sql_database.enterprise_ai.name
      }

      env {
        name  = "POSTGRES_USER"
        value = google_sql_user.enterprise_ai.name
      }

      env {
        name = "POSTGRES_PASSWORD"

        value_source {
          secret_key_ref {
            secret  = "deldai-enterprise-ai-dev-db-password"
            version = "1"
          }
        }
      }

      startup_probe {
        initial_delay_seconds = 5
        timeout_seconds       = 5
        period_seconds        = 10
        failure_threshold     = 6

        http_get {
          path = "/api/v1/health"
          port = 8000
        }
      }
    }

    vpc_access {
      egress = "PRIVATE_RANGES_ONLY"

      network_interfaces {
        network    = "default"
        subnetwork = "default"
      }
    }
  }

  depends_on = [
    google_secret_manager_secret_iam_member.control_plane_api_key_tenants_accessor,
    google_secret_manager_secret_iam_member.provider_credentials_accessor,
    google_secret_manager_secret_iam_member.db_password_accessor,
    google_secret_manager_secret_iam_member.api_key_accessor,
  ]
}
