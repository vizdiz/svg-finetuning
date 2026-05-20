variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "aws_profile" {
  type    = string
  default = ""
}

variable "project_name" {
  type    = string
  default = "svg-finetuning"
}

variable "hf_secret_id" {
  type    = string
  default = "svg-finetuning/huggingface-token"
}

variable "endpoint_name" {
  type    = string
  default = "svg-finetuning-inference"
}

variable "inference_endpoint_model" {
  type    = string
  default = "Qwen/Qwen2.5-7B-Instruct"
}

variable "inference_endpoint_mode" {
  type    = string
  default = "sagemaker"
}

variable "api_lambda_timeout_seconds" {
  type    = number
  default = 30
}

variable "api_lambda_memory_size" {
  type    = number
  default = 256
}

variable "training_image" {
  type    = string
  default = "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.8.0-transformers4.56.2-gpu-py312-cu129-ubuntu22.04"
}

variable "lambda_role_arn" {
  type    = string
  default = "arn:aws:iam::446224796301:role/SVGFinetuneLambdaRole"
}

variable "sagemaker_role_arn" {
  type    = string
  default = "arn:aws:iam::446224796301:role/SVGFinetuneSageMakerRole"
}

variable "weekly_cron" {
  type    = string
  default = "cron(0 2 ? * SUN *)"
}

variable "enable_s3_trigger" {
  type    = bool
  default = true
}

variable "enable_weekly_trigger" {
  type    = bool
  default = true
}

variable "lambda_timeout_seconds" {
  type    = number
  default = 300
}

variable "website_domain_name" {
  type    = string
  default = "svgen.net"
}

variable "website_www_domain_name" {
  type    = string
  default = "www.svgen.net"
}
