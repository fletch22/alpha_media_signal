
copy ..\..\..\requirements.txt .\requirements.txt

SET TAG="f22_airflow"

docker rmi %TAG% --force

SET DOCKER_BUILDKIT=1

docker build -t %TAG% .