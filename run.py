from pipelines import training
import click
from typing import Optional
from datetime import datetime as dt

@click.command()
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)

def main(
    no_cache: bool = False,
):
    """Main entry point for the pipeline execution.

    This function serves as the entry point for executing the pipeline. It allows you to
    specify whether caching should be enabled or disabled for the pipeline run.

    Args:
        no_cache (bool, optional): Whether to disable caching for the pipeline run.
            Defaults to False.

    Example:
        To run the pipeline with caching disabled, use the following command:
        ```
        main(no_cache=True)
        ```

    Returns:
        None
    """

    # This executes all steps in the pipeline in the correct order using the orchestrator
    # stack component that is configured in active ZenML stack.
    pipeline_args = {}
    if no_cache:
        pipeline_args["enable_cache"] = False
    train = training.training_pipeline()

if __name__ == "__main__":
    main()
