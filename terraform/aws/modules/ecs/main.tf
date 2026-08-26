resource "aws_ecs_cluster" "this" {
  name = "${var.project_name}-${var.environment}"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_cloudwatch_log_group" "gateway" {
  name              = "/ecs/${var.project_name}/${var.environment}/gateway"
  retention_in_days = 14

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_security_group" "ecs" {
  name        = "${var.project_name}-${var.environment}-ecs-sg"
  description = "Security group for ECS tasks"
  vpc_id      = var.vpc_id

  ingress {
    description     = "Gateway traffic from ALB"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [var.alb_security_group_id]
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_ecs_task_definition" "gateway" {
  family                   = "${var.project_name}-${var.environment}-gateway"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"

  cpu    = var.cpu
  memory = var.memory

  execution_role_arn = var.execution_role_arn
  task_role_arn      = var.task_role_arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  container_definitions = jsonencode([
    {
      name      = "llm-gateway"
      image     = var.container_image
      essential = true

      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "ENVIRONMENT"
          value = var.environment
        },

        {
          name  = "AWS_REGION"
          value = var.aws_region
        },

        {
          name  = "S3_BUCKET"
          value = var.s3_bucket_name
        },

        {
          name  = "OTEL_EXPORT_ENABLED"
          value = "false"
        },

        {
          name  = "OTEL_SERVICE_NAME"
          value = "llm-gateway"
        },

        {
          name  = "OTEL_SERVICE_VERSION"
          value = "1.0.0"
        },

        {
          name  = "OTEL_DEPLOYMENT_ENVIRONMENT"
          value = var.environment
        }
      ]

      secrets = [
        {
          name      = "PROVIDER_CREDENTIALS"
          valueFrom = var.provider_credentials_secret_arn
        },

        {
          name      = "PLATFORM_ENVIRONMENT"
          valueFrom = var.environment_parameter_arn
        },

        {
          name      = "LOG_LEVEL"
          valueFrom = var.log_level_parameter_arn
        },

        {
          name      = "DEFAULT_PROVIDER"
          valueFrom = var.default_provider_parameter_arn
        },

        {
          name      = "API_KEY"
          valueFrom = var.gateway_api_key_secret_arn
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"

        options = {
          awslogs-group         = aws_cloudwatch_log_group.gateway.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "gateway"
        }
      }

      healthCheck = {
        command = [
          "CMD-SHELL",
          "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=5)\" || exit 1"
        ]

        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 60
      }
    }
  ])

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_ecs_service" "gateway" {
  name            = "${var.project_name}-${var.environment}-gateway"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.gateway.arn

  desired_count = var.desired_count

  launch_type = "FARGATE"

  platform_version = "LATEST"

  network_configuration {
    subnets = var.public_subnet_ids

    security_groups = [
      aws_security_group.ecs.id
    ]

    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = var.target_group_arn
    container_name   = "llm-gateway"
    container_port   = 8000
  }

  health_check_grace_period_seconds = 120

  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 200

  depends_on = [
    var.alb_listener_dependency
  ]

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}
