from typing import Dict

API_KEY = 'api_key'
API_SECRET_KEY = 'api_secret_key'
ENDPOINT = 'endpoint'


class Credentials:
    def __init__(self, creds_yaml: Dict, cred_yaml_key: str):
        self.default_bearer_token = creds_yaml[cred_yaml_key]['bearer_token']

        twitter_cred = creds_yaml[cred_yaml_key]

        if API_KEY in twitter_cred:
            self.api_key = twitter_cred[API_KEY]

        if API_SECRET_KEY in twitter_cred:
            self.api_secret_key = twitter_cred[API_SECRET_KEY]

        if ENDPOINT in twitter_cred:
            self.endpoint = twitter_cred[ENDPOINT]
