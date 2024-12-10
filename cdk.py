import os
from pathlib import Path
from constructs import Construct
from aws_cdk import (
    App,
    Stack,
    Environment,
    Duration,
    CfnOutput,
)
from aws_cdk.aws_lambda import (
    DockerImageFunction,
    DockerImageCode,
    Architecture,
    FunctionUrlAuthType,
)
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
my_environment = Environment(
    account=os.environ["CDK_DEFAULT_ACCOUNT"], region=os.environ["CDK_DEFAULT_REGION"]
)


class MnistClassifierFastAPIStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create Lambda function
        lambda_fn = DockerImageFunction(
            self,
            "MnistFastAPI",
            code=DockerImageCode.from_image_asset(
                str(Path.cwd()),  # Uses the Dockerfile in the current directory
                file="Dockerfile",
            ),
            architecture=Architecture.X86_64,
            memory_size=10000,  # 10 GB
            timeout=Duration.minutes(15),
        )

        # Add HTTPS URL
        fn_url = lambda_fn.add_function_url(auth_type=FunctionUrlAuthType.NONE)

        # Output the function URL
        CfnOutput(
            self,
            "FunctionUrl",
            value=fn_url.url,
            description="URL for the Lambda function",
        )


app = App()
MnistClassifierFastAPIStack(app, "MnistStack", env=my_environment)
app.synth()
