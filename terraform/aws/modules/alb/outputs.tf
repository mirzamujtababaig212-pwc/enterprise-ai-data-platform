output "target_group_arn" {
  description = "ARN of the ALB target group."
  value       = aws_lb_target_group.gateway.arn
}

output "alb_security_group_id" {
  description = "Security group ID of the application load balancer."
  value       = aws_security_group.alb.id
}

output "http_listener" {
  description = "HTTP listener used by the gateway."
  value       = aws_lb_listener.http
}

output "alb_dns_name" {
  description = "DNS name of the application load balancer."
  value       = aws_lb.this.dns_name
}
