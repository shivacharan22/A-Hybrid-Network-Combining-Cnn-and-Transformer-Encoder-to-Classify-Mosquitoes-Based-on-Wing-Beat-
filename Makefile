
setup:
	pip install -r requirements.txt
	zenml integration install wandb -y
	zenml secret create wandb_secret --entity=<put your username> --project_name= <project_name> --api_key= <api_key>

install-stack:
	zenml experiment-tracker register wandb_tracker --flavor=wandb --entity={{wandb_secret.entity}} --project_name={{wandb_secret.project_name}} --api_key={{wandb_secret.api_key}} && \
	zenml stack register CNN_trans_stack -a default -o default -e wandb_tracker && \	
	zenml set stack CNN_trans_stack && \
	zenml stack list
	zenml stack up
