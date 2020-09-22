import json

from kafka import KafkaProducer

from ams.config import constants
from ams.services import Topic


class KafkaProducerService():
    def __init__(self, topic: Topic):
        self.producer = KafkaProducer(bootstrap_servers=[constants.KAFKA_URL],
                                      value_serializer=lambda x: json.dumps(x).encode('utf-8'))
        self.topic = topic

    def send_message(self, data: object):
        self.producer.send("", data)
