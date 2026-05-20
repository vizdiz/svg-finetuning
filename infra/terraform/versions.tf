terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.0"
    }
  }
}

provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile

  default_tags {
    tags = {
      Project = var.project_name
      Managed = "terraform"
    }
  }
}

provider "aws" {
  alias   = "no_default_tags"
  region  = var.aws_region
  profile = var.aws_profile
}

data "aws_caller_identity" "current" {}
