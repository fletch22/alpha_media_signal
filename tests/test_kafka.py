from json import dumps

from kafka import KafkaConsumer
from kafka import KafkaProducer

from ams.services.KafkaProducerService import KafkaProducerService
from ams.services.Topic import Topic
from ams.utils.PrinterThread import PrinterThread


def test_kafka():
    # Arrange
    topic = "f22-foo"
    kafka_prod_svc = KafkaProducerService(topic=Topic.RAW_DROP)

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers='localhost:9092',
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='f22-alpha-media-signal',
        # value_deserializer=lambda x: loads(x.decode('utf-8'))),
        value_deserializer=lambda x: x)

    # Act
    for e in range(3):
        data = {'number': e}
        kafka_prod_svc.send_message(data=data)

    pt = PrinterThread()
    try:
        pt.start()
        for message in consumer:
            pt.print('Got message!')
            # message = message.value
            pt.print(f'Consumed message {message} from topic {topic}')
            # time.sleep(10)
            # break
    finally:
        pt.end()

    # Assert


if __name__ == '__main__':
    test_kafka()
