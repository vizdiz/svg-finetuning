output "data_bucket" {
  value = aws_s3_bucket.data.bucket
}

output "models_bucket" {
  value = aws_s3_bucket.models.bucket
}

output "scripts_bucket" {
  value = aws_s3_bucket.scripts.bucket
}

output "logs_bucket" {
  value = aws_s3_bucket.logs.bucket
}

output "lambda_name" {
  value = aws_lambda_function.retraining_trigger.function_name
}

output "lambda_role_arn" {
  value = var.lambda_role_arn
}

output "sagemaker_role_arn" {
  value = var.sagemaker_role_arn
}

output "website_bucket" {
  value = aws_s3_bucket.website.bucket
}

output "website_url" {
  value = "https://${aws_cloudfront_distribution.website.domain_name}"
}

output "website_certificate_arn" {
  value = aws_acm_certificate.website.arn
}

output "website_certificate_validation_records" {
  value = [
    for record in aws_acm_certificate.website.domain_validation_options : {
      domain_name           = record.domain_name
      resource_record_name  = record.resource_record_name
      resource_record_type  = record.resource_record_type
      resource_record_value = record.resource_record_value
    }
  ]
}

output "api_url" {
  value = "https://${aws_cloudfront_distribution.website.domain_name}/api"
}

output "sessions_table" {
  value = aws_dynamodb_table.sessions.name
}
