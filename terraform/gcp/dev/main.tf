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
