SET AIRFLOW_HOME=%HOME%\airflow

airflow db init

airflow users create \
    --username admin \
    --firstname Chris \
    --lastname Flesche \
    --role Admin \
    --email chris@fletch22.com

# start the web server, default port is 8080
airflow webserver --port 8080

# start the scheduler
# open a new terminal or else run webserver with ``-D`` option to run it as a daemon
airflow scheduler

# visit localhost:8080 in the browser and use the admin account you just
# created to login. Enable the example_bash_operator dag in the home page