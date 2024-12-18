---
title: My Gradio App Mnist Classifier
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.7.1"
app_file: app.py
pinned: false
---

# aws_ec2_automation
Here’s a detailed explanation of the GitHub Actions (GHA) pipeline in **raw Markdown format**:

---

# GitHub Actions Pipeline Documentation

## Name: Deploy PyTorch Training with EC2 Runner and Docker Compose

This pipeline automates the following tasks:
1. Starts an EC2 instance as a self-hosted GitHub runner.
2. Deploys a PyTorch training pipeline using Docker Compose.
3. Builds, tags, and pushes Docker images to Amazon ECR.
4. Stops the EC2 instance after the job is completed.

---

### Workflow Triggers

```yaml
on:
  push:
    branches:
      - main
```

- **Trigger**: This workflow runs whenever a push is made to the `main` branch.

---

## Jobs Overview

### 1. **start-runner**
Starts a self-hosted EC2 runner using the GitHub Actions Runner.

#### Steps:
1. **Configure AWS Credentials**:
   ```yaml
   - name: Configure AWS credentials
     uses: aws-actions/configure-aws-credentials@v4
     with:
       aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
       aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
       aws-region: ${{ secrets.AWS_REGION }}
   ```
   - Authenticates with AWS using access keys and the region specified in the secrets.
   - Required for creating and managing the EC2 instance.

2. **Start EC2 Runner**:
   ```yaml
   - name: Start EC2 runner
     id: start-ec2-runner
     uses: machulav/ec2-github-runner@v2
     with:
       mode: start
       github-token: ${{ secrets.GH_PERSONAL_ACCESS_TOKEN }}
       ec2-image-id: ami-044b0717aadbc9dfa
       ec2-instance-type: t2.xlarge
       subnet-id: subnet-024811dee81325f1c
       security-group-id: sg-0646c2a337a355a31
   ```
   - Starts an EC2 instance with the specified AMI, instance type, subnet, and security group.
   - Outputs:
     - `label`: A unique label for the EC2 runner.
     - `ec2-instance-id`: The ID of the created EC2 instance.

---

### 2. **deploy**
Deploys the PyTorch training pipeline using the EC2 runner started in the previous step.

#### Dependencies:
```yaml
needs: start-runner
runs-on: ${{ needs.start-runner.outputs.label }}
```
- **Depends on** the `start-runner` job and runs on the newly created EC2 instance.

#### Steps:
1. **Checkout Repository**:
   ```yaml
   - name: Checkout repository
     uses: actions/checkout@v4
   ```
   - Clones the current repository to the runner.

2. **Set Up Docker Buildx**:
   ```yaml
   - name: Set up Docker Buildx
     uses: docker/setup-buildx-action@v3
   ```
   - Configures Docker Buildx for building multi-platform Docker images.

3. **Configure AWS Credentials**:
   ```yaml
   - name: Configure AWS credentials
     uses: aws-actions/configure-aws-credentials@v4
     with:
       aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
       aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
       aws-region: ${{ secrets.AWS_REGION }}
   ```
   - Reconfigures AWS credentials for Docker ECR authentication and resource management.

4. **Log in to Amazon ECR**:
   ```yaml
   - name: Log in to Amazon ECR
     id: login-ecr
     uses: aws-actions/amazon-ecr-login@v2
   ```
   - Logs into Amazon ECR for pushing and pulling Docker images.

5. **Create `.env` File**:
   ```yaml
   - name: Create .env file
     run: |
       echo "AWS_ACCESS_KEY_ID=${{ secrets.AWS_ACCESS_KEY_ID }}" >> .env
       echo "AWS_SECRET_ACCESS_KEY=${{ secrets.AWS_SECRET_ACCESS_KEY }}" >> .env
       echo "AWS_REGION=${{ secrets.AWS_REGION }}" >> .env
   ```
   - Generates a `.env` file for the application with AWS credentials and region.

6. **Run Docker Compose for Train and Eval Services**:
   ```yaml
   - name: Run Docker Compose for train and eval service
     run: |
       docker-compose build
       docker-compose up --build
       docker-compose logs --follow
       docker-compose down --remove-orphans
   ```
   - **Build**: Builds all services defined in the `docker-compose.yml` file.
   - **Up**: Runs all services, including training and evaluation.
   - **Logs**: Outputs logs for debugging purposes.
   - **Down**: Stops all services and removes orphaned containers.

7. **Build, Tag, and Push Docker Image to Amazon ECR**:
   ```yaml
   - name: Build, tag, and push Docker image to Amazon ECR
     env:
       REGISTRY: ${{ steps.login-ecr.outputs.registry }}
       REPOSITORY: soutrik71/mnist
       IMAGE_TAG: ${{ github.sha }}
     run: |
       docker build -t $REGISTRY/$REPOSITORY:$IMAGE_TAG .
       docker push $REGISTRY/$REPOSITORY:$IMAGE_TAG
       docker tag $REGISTRY/$REPOSITORY:$IMAGE_TAG $REGISTRY/$REPOSITORY:latest
       docker push $REGISTRY/$REPOSITORY:latest
   ```
   - **Build**: Creates a Docker image with the repository and tag.
   - **Push**: Pushes the image to Amazon ECR.
   - **Tag**: Updates the `latest` tag.

8. **Pull and Verify Docker Image from ECR**:
   ```yaml
   - name: Pull Docker image from ECR and verify
     env:
       REGISTRY: ${{ steps.login-ecr.outputs.registry }}
       REPOSITORY: soutrik71/mnist
       IMAGE_TAG: ${{ github.sha }}
     run: |
       docker pull $REGISTRY/$REPOSITORY:$IMAGE_TAG
       docker images | grep "$REGISTRY/$REPOSITORY"
   ```
   - **Pull**: Pulls the built image from ECR.
   - **Verify**: Ensures the image exists locally.

9. **Clean Up Environment**:
   ```yaml
   - name: Clean up environment
     run: |
       rm -f .env
       docker system prune -af
   ```
   - Deletes the `.env` file and removes unused Docker resources.

---

### 3. **stop-runner**
Stops and terminates the EC2 runner created in the `start-runner` job.

#### Dependencies:
```yaml
needs:
  - start-runner
  - deploy
```

#### Steps:
1. **Configure AWS Credentials**:
   ```yaml
   - name: Configure AWS credentials
     uses: aws-actions/configure-aws-credentials@v4
     with:
       aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
       aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
       aws-region: ${{ secrets.AWS_REGION }}
   ```

2. **Stop EC2 Runner**:
   ```yaml
   - name: Stop EC2 runner
     uses: machulav/ec2-github-runner@v2
     with:
       mode: stop
       github-token: ${{ secrets.GH_PERSONAL_ACCESS_TOKEN }}
       label: ${{ needs.start-runner.outputs.label }}
       ec2-instance-id: ${{ needs.start-runner.outputs.ec2-instance-id }}
   ```
   - Stops the EC2 runner instance created in the first job.

3. **Validate EC2 Termination**:
   ```yaml
   - name: Validate EC2 termination
     run: aws ec2 describe-instances --instance-ids ${{ needs.start-runner.outputs.ec2-instance-id }}
   ```
   - Ensures the EC2 instance has been properly terminated.

---

### Key Highlights
1. **Sequential Execution**:
   - The `start-runner`, `deploy`, and `stop-runner` jobs are executed sequentially.

2. **Error Handling**:
   - The `stop-runner` job runs even if previous jobs fail (`if: ${{ always() }}`).

3. **Efficiency**:
   - Docker layer caching speeds up builds.
   - Cleanup steps maintain a clean environment.

4. **Security**:
   - Secrets are masked and removed after use.
   - Proper resource cleanup ensures cost efficiency.

---

This pipeline ensures robust deployment with error handling, logging, and cleanup mechanisms. So far we have discussed the GitHub Actions pipeline , the basic structure of the pipeline, and the steps involved in the pipeline.
Next we will have an interdependent pipeline where the output of one job will be used as input for the next job.

---
## Advanced Pipeline with 
* Sequential Flow: Each job has clear dependencies, ensuring no step runs out of order.
* Code Checkout: Explicit repository checkout in each job ensures consistent source code.
* Secure Credential Handling: Sensitive credentials are masked and stored securely.
* Resource Cleanup: Includes Docker clean-up and EC2 instance termination validation.
* Logging: Added detailed logs to improve debugging and monitoring.


Step 1: Start EC2 Runner
  Purpose: Initializes a self-hosted EC2 runner for running subsequent jobs.
  Key Actions:
  Configures AWS credentials.
  Launches an EC2 instance using specified AMI, instance type, and networking configurations.
  Outputs the runner label and instance ID for downstream jobs.
Step 2: Test PyTorch Code Using Docker Compose
  Purpose: Tests the PyTorch training and evaluation services.
  Key Actions:
  Checks out the repository.
  Sets up Docker Buildx for advanced build capabilities.
  Configures AWS credentials and creates a masked .env file for secure credential sharing.
  Runs all services (train, eval) using Docker Compose, monitors logs, and cleans up containers.
Step 3: Build, Tag, and Push Docker Image
  Purpose: Builds a Docker image, tags it, and pushes it to Amazon ECR after successful tests.
  Key Actions:
  Checks out the repository again to ensure consistency.
  Logs into Amazon ECR using AWS credentials.
  Builds and tags the Docker image with latest and SHA-based tags.
  Pushes the image to Amazon ECR and verifies by pulling it back.
Step 4: Stop and Delete EC2 Runner
  Purpose: Stops and terminates the EC2 instance to ensure cost efficiency and cleanup.
  Key Actions:
  Configures AWS credentials.
  Stops the EC2 instance using the label and instance ID from start-runner.
  Validates the termination state of the EC2 instance to ensure proper cleanup.



#### FastAPI and Aws Serverless deployment strategy from aws machine
```bash
# check the venv with poetry
poetry lock
poetry install --no-root

# run in local
python app.py

# create a docker image and run
docker build -t fastapi-aws-mnist:latest .
docker run --name fastapi-aws-mnist-container -p 8000:8000 --env-file .env fastapi-aws-mnist:latest
docker stop fastapi-aws-mnist-container
docker rm -f fastapi-aws-mnist-container
docker run -it --rm -p 8000:8000 --env-file .env fastapi-aws-mnist:latest bash # to run bash in the container to check the env variables

# serverless deployment pre install aws cdk and aws cli
npm install -g aws-cdk

# convert the fastapi code with UI using fasthtml and Shad4FastHtml
# create a main.py file where we modify the fastapi code to include the UI and check the docker file too
uvicorn main:app --host 0.0.0.0 --port 8000

```

#### If deployed from non aws machine

```bash
# first install aws cli and configure the aws cli
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
aws --version
aws configure
aws configure list
aws s3 ls
# in env local and GHA have the env variables like AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, CDK_DEFAULT_ACCOUNT, CDK_DEFAULT_REGION
```

#### AWS CDK Integration
```bash
  # npm install -g aws-cdk
  # In cdk.py file CatDogClassifierFastAPIStack class is used to fetch a Dockerfile and create a container like service with aws lambda service.
  # While developing in local make sure that you have set all the environment variables
  # aws configure -> AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION
  # CDK_DEFAULT_ACCOUNT, CDK_DEFAULT_REGION
  # test the docker image locally
  cdk bootstrap -v # command is used to set up the necessary AWS resources that AWS CDK needs to deploy your infrastructure. It prepares your AWS environment for the first time deployment or when you need to deploy infrastructure that requires certain resources like storing docker image, I am roles, CloudFormation.
  cdk deploy -v --logs # Deploys your infrastructure to AWS as defined by your CDK stack.
  cdk destroy -vf #Destroy all resources created about without any traces left.**
  # Connectivity with another S3 bucker doesnot work as the bucket is not public and the bucket policy is not set to public.
  # all loggers should be readonly so that while deploying the stack the logs doesnot get deleted.





