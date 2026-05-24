locals {
  account_id      = data.aws_caller_identity.current.account_id
  data_bucket     = "${var.project_name}-data-${local.account_id}"
  models_bucket   = "${var.project_name}-models-${local.account_id}"
  scripts_bucket  = "${var.project_name}-scripts-${local.account_id}"
  logs_bucket     = "${var.project_name}-logs-${local.account_id}"
  lambda_name     = "${var.project_name}-retraining-trigger"
  api_lambda_name = "${var.project_name}-api-gateway"
  sessions_table  = "${var.project_name}-sessions-${local.account_id}"
  website_bucket  = "${var.project_name}-website-${local.account_id}"
  api_origin_id   = "${var.project_name}-api-origin"
}

resource "aws_s3_bucket" "data" {
  bucket        = local.data_bucket
  force_destroy = true
}

resource "aws_s3_bucket" "models" {
  bucket        = local.models_bucket
  force_destroy = true
}

resource "aws_s3_bucket" "scripts" {
  bucket        = local.scripts_bucket
  force_destroy = true
}

resource "aws_s3_bucket" "logs" {
  bucket        = local.logs_bucket
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket                  = aws_s3_bucket.models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "scripts" {
  bucket                  = aws_s3_bucket.scripts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "logs" {
  bucket                  = aws_s3_bucket.logs.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "scripts" {
  bucket = aws_s3_bucket.scripts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "logs" {
  bucket = aws_s3_bucket.logs.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "scripts" {
  bucket = aws_s3_bucket.scripts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

data "archive_file" "lambda_zip" {
  type        = "zip"
  source_file = "${path.module}/../../lambda/lambda_function.py"
  output_path = "${path.module}/.build/lambda_function.zip"
}

resource "aws_lambda_function" "retraining_trigger" {
  function_name    = local.lambda_name
  role             = var.lambda_role_arn
  handler          = "lambda_function.handler"
  runtime          = "python3.12"
  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  timeout          = var.lambda_timeout_seconds
  memory_size      = 256

  environment {
    variables = {
      ACCOUNT_ID     = local.account_id
      DATA_BUCKET    = aws_s3_bucket.data.bucket
      MODELS_BUCKET  = aws_s3_bucket.models.bucket
      SCRIPTS_URI    = "s3://${aws_s3_bucket.scripts.bucket}/training/"
      SM_ROLE        = var.sagemaker_role_arn
      ENDPOINT_NAME  = var.endpoint_name
      TRAINING_IMAGE = var.training_image
      HF_SECRET_ID   = var.hf_secret_id
    }
  }
}

resource "aws_dynamodb_table" "sessions" {
  name         = local.sessions_table
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "session_id"

  attribute {
    name = "session_id"
    type = "S"
  }
}

data "aws_iam_policy_document" "api_lambda_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "api_lambda" {
  name               = "${var.project_name}-api-lambda-role"
  assume_role_policy = data.aws_iam_policy_document.api_lambda_assume_role.json
}

data "aws_iam_policy_document" "api_lambda_policy" {
  statement {
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["*"]
  }

  statement {
    effect = "Allow"
    actions = [
      "dynamodb:GetItem",
      "dynamodb:PutItem",
    ]
    resources = [aws_dynamodb_table.sessions.arn]
  }

  statement {
    effect = "Allow"
    actions = [
      "sagemaker:InvokeEndpoint",
    ]
    resources = ["arn:aws:sagemaker:${var.aws_region}:${local.account_id}:endpoint/${var.endpoint_name}"]
  }

  statement {
    effect = "Allow"
    actions = [
      "bedrock:InvokeModel",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "api_lambda" {
  name   = "${var.project_name}-api-lambda-policy"
  role   = aws_iam_role.api_lambda.id
  policy = data.aws_iam_policy_document.api_lambda_policy.json
}

data "archive_file" "api_lambda_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../../"
  output_path = "${path.module}/.build/api_lambda.zip"
  excludes = [
    ".git/**",
    ".venv/**",
    "node_modules/**",
    "dist/**",
    "pipeline_output/**",
    "scratch/**",
    "website/**",
    "tests/**",
    ".cache/**",
    "infra/**",
    "infra/.terraform/**",
    "infra/terraform/.build/**",
    "infra/terraform/terraform.tfstate*",
    "__pycache__/**",
    ".pytest_cache/**",
    ".DS_Store",
  ]
}

resource "aws_lambda_function" "api" {
  function_name    = local.api_lambda_name
  role             = aws_iam_role.api_lambda.arn
  handler          = "backend.api_handler.handler"
  runtime          = "python3.12"
  filename         = data.archive_file.api_lambda_zip.output_path
  source_code_hash = data.archive_file.api_lambda_zip.output_base64sha256
  timeout          = var.api_lambda_timeout_seconds
  memory_size      = var.api_lambda_memory_size

  environment {
    variables = {
      SESSION_STORE_MODE = "dynamodb"
      SESSION_TABLE_NAME = aws_dynamodb_table.sessions.name
      ENDPOINT_MODE      = var.inference_endpoint_mode
      ENDPOINT_NAME      = var.endpoint_name
      ENDPOINT_MODEL     = var.inference_endpoint_model
      ENDPOINT_TIMEOUT_S = "60"
      CACHE_TTL_SECONDS  = "86400"
      BEDROCK_REPAIR_ENABLED  = "true"
      BEDROCK_REPAIR_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
    }
  }
}

resource "aws_apigatewayv2_api" "api" {
  name          = "${var.project_name}-api"
  protocol_type = "HTTP"

  cors_configuration {
    allow_headers = ["content-type", "authorization"]
    allow_methods = ["GET", "POST", "OPTIONS"]
    allow_origins = ["*"]
  }
}

resource "aws_apigatewayv2_integration" "api" {
  api_id                 = aws_apigatewayv2_api.api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.api.invoke_arn
  integration_method     = "POST"
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "api" {
  api_id    = aws_apigatewayv2_api.api.id
  route_key = "ANY /api/{proxy+}"
  target    = "integrations/${aws_apigatewayv2_integration.api.id}"
}

resource "aws_apigatewayv2_route" "api_root" {
  api_id    = aws_apigatewayv2_api.api.id
  route_key = "ANY /api"
  target    = "integrations/${aws_apigatewayv2_integration.api.id}"
}

resource "aws_apigatewayv2_stage" "api" {
  api_id      = aws_apigatewayv2_api.api.id
  name        = "$default"
  auto_deploy = true
}

resource "aws_lambda_permission" "allow_apigw_api" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.api.execution_arn}/*/*"
}

resource "aws_cloudfront_cache_policy" "api_no_cache" {
  provider    = aws.no_default_tags
  name        = "${var.project_name}-api-no-cache"
  comment     = "No-cache policy for API requests"
  default_ttl = 0
  max_ttl     = 0
  min_ttl     = 0

  parameters_in_cache_key_and_forwarded_to_origin {
    cookies_config {
      cookie_behavior = "none"
    }

    headers_config {
      header_behavior = "none"
    }

    query_strings_config {
      query_string_behavior = "none"
    }
  }
}

resource "aws_cloudfront_origin_request_policy" "api_all_viewer" {
  provider = aws.no_default_tags
  name     = "${var.project_name}-api-all-viewer"
  comment  = "Forward all viewer inputs to the API"

  cookies_config {
    cookie_behavior = "all"
  }

  headers_config {
    header_behavior = "allViewer"
  }

  query_strings_config {
    query_string_behavior = "all"
  }
}

resource "aws_lambda_permission" "allow_s3" {
  count         = var.enable_s3_trigger ? 1 : 0
  statement_id  = "AllowExecutionFromS3"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.retraining_trigger.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.data.arn
}

resource "aws_s3_bucket_notification" "data_manifest_trigger" {
  count  = var.enable_s3_trigger ? 1 : 0
  bucket = aws_s3_bucket.data.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.retraining_trigger.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "train/"
    filter_suffix       = "dataset_manifest.json"
  }

  depends_on = [aws_lambda_permission.allow_s3]
}

resource "aws_cloudwatch_event_rule" "weekly_retrain" {
  count               = var.enable_weekly_trigger ? 1 : 0
  name                = "${var.project_name}-weekly-retrain"
  schedule_expression = var.weekly_cron
}

resource "aws_cloudwatch_event_target" "weekly_retrain" {
  count     = var.enable_weekly_trigger ? 1 : 0
  target_id = "weekly-retrain"
  rule      = aws_cloudwatch_event_rule.weekly_retrain[0].name
  arn       = aws_lambda_function.retraining_trigger.arn
}

resource "aws_lambda_permission" "allow_eventbridge" {
  count         = var.enable_weekly_trigger ? 1 : 0
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.retraining_trigger.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.weekly_retrain[0].arn
}

resource "aws_s3_bucket" "website" {
  bucket        = local.website_bucket
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "website" {
  bucket                  = aws_s3_bucket.website.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "website" {
  bucket = aws_s3_bucket.website.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "website" {
  bucket = aws_s3_bucket.website.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_cloudfront_origin_access_control" "website" {
  provider                          = aws.no_default_tags
  name                              = "${var.project_name}-website-oac"
  description                       = "Origin access control for ${var.project_name} website"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

data "aws_iam_policy_document" "website_bucket" {
  statement {
    sid    = "AllowCloudFrontRead"
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["cloudfront.amazonaws.com"]
    }

    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.website.arn}/*"]

    condition {
      test     = "StringEquals"
      variable = "AWS:SourceArn"
      values   = [aws_cloudfront_distribution.website.arn]
    }
  }
}

resource "aws_s3_bucket_policy" "website" {
  bucket     = aws_s3_bucket.website.id
  policy     = data.aws_iam_policy_document.website_bucket.json
  depends_on = [aws_s3_bucket_public_access_block.website]
}

resource "aws_acm_certificate" "website" {
  provider                  = aws.no_default_tags
  domain_name               = var.website_domain_name
  subject_alternative_names = [var.website_www_domain_name]
  validation_method         = "DNS"

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_cloudfront_distribution" "website" {
  provider            = aws.no_default_tags
  enabled             = true
  comment             = "${var.project_name} static site"
  default_root_object = "index.html"
  price_class         = "PriceClass_100"
  aliases             = [var.website_domain_name, var.website_www_domain_name]

  origin {
    domain_name              = aws_s3_bucket.website.bucket_regional_domain_name
    origin_id                = local.website_bucket
    origin_access_control_id = aws_cloudfront_origin_access_control.website.id
  }

  origin {
    domain_name = replace(aws_apigatewayv2_api.api.api_endpoint, "https://", "")
    origin_id   = local.api_origin_id
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    target_origin_id       = local.website_bucket
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD", "OPTIONS"]
    cached_methods         = ["GET", "HEAD", "OPTIONS"]
    compress               = true
    cache_policy_id        = "658327ea-f89d-4fab-a63d-7e88639e58f6"
  }

  ordered_cache_behavior {
    path_pattern             = "api/*"
    target_origin_id         = local.api_origin_id
    viewer_protocol_policy   = "redirect-to-https"
    allowed_methods          = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods           = ["GET", "HEAD", "OPTIONS"]
    compress                 = true
    cache_policy_id          = aws_cloudfront_cache_policy.api_no_cache.id
    origin_request_policy_id = aws_cloudfront_origin_request_policy.api_all_viewer.id
  }

  custom_error_response {
    error_code            = 403
    response_code         = 200
    response_page_path    = "/index.html"
    error_caching_min_ttl = 0
  }

  custom_error_response {
    error_code            = 404
    response_code         = 200
    response_page_path    = "/index.html"
    error_caching_min_ttl = 0
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate.website.arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }
}
