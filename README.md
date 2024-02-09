# evaluation
Step 1: Create a new virtual environment
	python3 -m venv .env
	source .env/bin/activate
	
Step 2: Clone this repository and install the requirements from requirements.txt
	git clone https://github.com/zuhaqm/evaluation.git
	pip install -r requirements.txt
	
Step 3: Train model
	Give the path of the dataset in the dataloader.py in the dataset folder file and set the train test split ratio.
	Run train.py in the tools folder adjusting the hyperparameters according to your requirements. The training loss is plotted with wandb

Step 4: Test model
	After the training is done, run test.py. Testing accuracy is plotted with wandb
	
Step 5: fastAPI, Dockerizing and Gradio app
	Build the dockerized api by stepping into the fastapi-docker-ml folder and run following command:
	docker build -t fastapi-docker-ml 
	Run the docker container with the following command:
	docker run -d -p 8000:8000 fastapi-docker-ml
	
Step 6: Infer with gradio app by running the gradio_app.py file.



