"""Publish fake shipment data to a Kafka topic."""

import json
import random
import time
import uuid
from datetime import datetime, timezone

from kafka import KafkaProducer

TOPIC = "shipments"
BROKER = "localhost:9092"

CARRIERS = ["FedEx", "UPS", "USPS", "DHL", "Amazon Logistics"]
STATUSES = [
    "label_created",
    "picked_up",
    "in_transit",
    "out_for_delivery",
    "delivered",
    "returned",
]
ORIGINS = [
    "New York, NY",
    "Los Angeles, CA",
    "Chicago, IL",
    "Houston, TX",
    "Phoenix, AZ",
    "Dallas, TX",
    "Atlanta, GA",
    "Seattle, WA",
]
DESTINATIONS = [
    "Miami, FL",
    "Denver, CO",
    "Boston, MA",
    "Portland, OR",
    "Nashville, TN",
    "San Diego, CA",
    "Minneapolis, MN",
    "Detroit, MI",
]


def _random_shipment() -> dict:
    """Generate a single random shipment record."""
    return {
        "shipment_id": str(uuid.uuid4()),
        "order_id": f"ORD-{random.randint(100000, 999999)}",
        "carrier": random.choice(CARRIERS),
        "status": random.choice(STATUSES),
        "origin": random.choice(ORIGINS),
        "destination": random.choice(DESTINATIONS),
        "weight_kg": round(random.uniform(0.5, 30.0), 2),
        "estimated_delivery": (
            datetime.now(timezone.utc)
            .replace(
                day=random.randint(1, 28),
                hour=random.randint(8, 20),
            )
            .isoformat()
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    producer = KafkaProducer(
        bootstrap_servers=BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    print(f"Publishing shipment messages to topic '{TOPIC}' on {BROKER}")
    print("Press Ctrl+C to stop.\n")

    try:
        count = 0
        while True:
            shipment = _random_shipment()
            producer.send(TOPIC, value=shipment)
            count += 1
            print(f"[{count}] Sent shipment {shipment['shipment_id']}  "
                  f"status={shipment['status']}  carrier={shipment['carrier']}")
            time.sleep(random.uniform(0.5, 2.0))
    except KeyboardInterrupt:
        print(f"\nStopped. {count} messages sent.")
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()
