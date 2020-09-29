

md tmp

copy ..\..\pip_requirements.txt .\tmp\

SET TAG="f22-spark-notebook"

docker rmi %TAG%

docker build -f Dockerfile . -t %TAG%

rmdir /Q /S .\tmp
