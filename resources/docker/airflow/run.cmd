REM you must run docker-compose command from folder with docker-
REM compose file (main_folder) in our case, this is not mandatory,
REM just to simplify tutorial

REM To re-initialize the job database (nuke and pave)
REM docker-compose up --build postgres
REM docker-compose up --build initdb

docker-compose up

