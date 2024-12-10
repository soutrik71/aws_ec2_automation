from constructs import Construct
from aws_cdk import (
    Duration,
    Stack,
    aws_iam as iam,
    aws_sqs as sqs,
    aws_sns as sns,
    aws_sns_subscriptions as subs,
)


class MyCdkAppStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        queue = sqs.Queue(
            self, "MyCdkAppQueue",
            visibility_timeout=Duration.seconds(300),
        )

        topic = sns.Topic(
            self, "MyCdkAppTopic"
        )

        topic.add_subscription(subs.SqsSubscription(queue))
