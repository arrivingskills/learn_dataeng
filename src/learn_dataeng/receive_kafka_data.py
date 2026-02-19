"""Subscribe to the shipments Kafka topic and print messages."""

import json

from kafka import KafkaConsumer

TOPIC = "shipments"
BROKER = "localhost:9092"


def main() -> None:
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BROKER,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="shipment-printer",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    print(f"Listening on topic '{TOPIC}' ({BROKER})…")
    print("Press Ctrl+C to stop.\n")

    try:
        for message in consumer:
            shipment = message.value
            print(
                f"[offset {message.offset}] "
                f"shipment={shipment['shipment_id']}  "
                f"status={shipment['status']}  "
                f"carrier={shipment['carrier']}  "
                f"{shipment['origin']} → {shipment['destination']}"
            )
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
