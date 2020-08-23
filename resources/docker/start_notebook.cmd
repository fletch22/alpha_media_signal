cls


set ENVIRONMENT="bind"

cd ..\..\

docker-compose -f resources\docker\spark_notebook.yaml up

cd resources\docker

