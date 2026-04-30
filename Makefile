.PHONY: help install test test-fast lint fmt sync \
        local-invoke build push deploy invoke logs status \
        clean

# AWS / infra config — read from terraform outputs in tcc_iac/infra
TF_DIR        := ../tcc_iac/infra
AWS_REGION    := $(shell terraform -chdir=$(TF_DIR) output -raw aws_region 2>/dev/null || echo us-east-1)
AWS_ACCOUNT   := 207121581830
ECR_REPO      := $(shell terraform -chdir=$(TF_DIR) output -raw ecr_repository_url 2>/dev/null || echo $(AWS_ACCOUNT).dkr.ecr.$(AWS_REGION).amazonaws.com/tcc-regime-etl)
LAMBDA_FN     := tcc-regime-etl
IMAGE_TAG     ?= latest
IMAGE_URI     := $(ECR_REPO):$(IMAGE_TAG)

help:
	@echo "Targets:"
	@echo "  install       uv sync"
	@echo "  test          uv run task test"
	@echo "  test-fast     uv run task test-fast"
	@echo "  lint / fmt    ruff check / format"
	@echo "  local-invoke  SAM local invocation (no AWS)"
	@echo "  build         docker buildx build --platform linux/arm64 → $(IMAGE_URI)"
	@echo "  push          ECR login + docker push"
	@echo "  deploy        build + push + lambda update-function-code"
	@echo "  invoke        aws lambda invoke (sync) and print response"
	@echo "  logs          tail CloudWatch logs"
	@echo "  status        show Lambda config + last modified"

install:
	uv sync

test:
	uv run task test

test-fast:
	uv run task test-fast

lint:
	uv run task lint

fmt:
	uv run task fmt

local-invoke:
	uv run task sam-invoke

build:
	docker buildx build --platform linux/arm64 --provenance=false \
	    -f docker/Dockerfile -t $(IMAGE_URI) --load .

push:
	aws ecr get-login-password --region $(AWS_REGION) \
	    | docker login --username AWS --password-stdin $(AWS_ACCOUNT).dkr.ecr.$(AWS_REGION).amazonaws.com
	docker push $(IMAGE_URI)

deploy: build push
	aws lambda update-function-code \
	    --function-name $(LAMBDA_FN) \
	    --image-uri $(IMAGE_URI) \
	    --region $(AWS_REGION) \
	    --no-cli-pager
	aws lambda wait function-updated \
	    --function-name $(LAMBDA_FN) --region $(AWS_REGION)
	@echo "Lambda updated."

invoke:
	@mkdir -p /tmp/tcc-etl
	aws lambda invoke \
	    --function-name $(LAMBDA_FN) \
	    --region $(AWS_REGION) \
	    --cli-binary-format raw-in-base64-out \
	    --payload '{}' \
	    --no-cli-pager \
	    /tmp/tcc-etl/response.json
	@echo "--- response.json ---"
	@cat /tmp/tcc-etl/response.json
	@echo

logs:
	aws logs tail /aws/lambda/$(LAMBDA_FN) --follow --region $(AWS_REGION)

status:
	aws lambda get-function-configuration \
	    --function-name $(LAMBDA_FN) --region $(AWS_REGION) --no-cli-pager \
	    --query '{Memory:MemorySize,Timeout:Timeout,Image:Code.ImageUri,LastModified:LastModified,State:State,LastUpdate:LastUpdateStatus}'

clean:
	rm -rf .aws-sam .pytest_cache .ruff_cache build dist
