# API

This folder contains a demo web server for serving generated sequences via REST API.

## Running the Server

Before you can use the server you need to get finetuned weights for the model and put them in `text_generation/weights`.

To run the server on the local machine execute this command (you might need to add the path to `text_generation` for `PYTHONPATH`):

```sh
python api/app.py
```

### Using Docker

You can also launch the server in a Docker container. To do that you first need to build the image:

```sh
docker build -t text_generation_api -f Dockerfile .
```

After that you need to launch a container (2 cpu cores and 1GB of RAM should be enough to quickly generate 1 sequence at a time):

```sh
docker run -p 8080:8080 --name api -it --rm  --cpus 2 --memory 1GB text_generation_api
```

### Getting a response

To test the server on a local machine you can execute this command:

```
curl --header "Content-Type: application/json" --request POST http://127.0.0.1:8080/generate -d '{"n_sequences": 1}'
```
