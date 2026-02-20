"""Subscribe to the shipments Kafka topic and write RDF (Turtle) files.

Each incoming shipment message is converted into an RDF graph that models
Shipment, Order, Carrier and Location as classes.  Status is constrained to
the named individuals ``Booked``, ``Shipped`` and ``Delivered``.
"""

import json
import re
from pathlib import Path

from kafka import KafkaConsumer
from rdflib import Graph, Literal, Namespace, RDF, RDFS, XSD

TOPIC = "shipments"
BROKER = "localhost:9092"
OUTPUT_DIR = Path("ttl_output")

# ── Namespaces ───────────────────────────────────────────────────────────────
SHIP = Namespace("http://arrivingskillsai.com/shipment#")

# ── Status mapping ───────────────────────────────────────────────────────────
# The producer emits varied statuses; we normalise them to three allowed values.
_STATUS_MAP: dict[str, str] = {
    "label_created": "Booked",
    "picked_up": "Shipped",
    "in_transit": "Shipped",
    "out_for_delivery": "Shipped",
    "delivered": "Delivered",
    "returned": "Booked",
    # also accept the three canonical names directly
    "booked": "Booked",
    "shipped": "Shipped",
}


def _slug(text: str) -> str:
    """Turn a human string into a URI-safe slug (e.g. 'New York, NY' → 'New_York_NY')."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def _build_graph(shipment: dict) -> Graph:
    """Convert a single shipment dict into an RDF graph."""
    g = Graph()
    g.bind("ship", SHIP)
    g.bind("xsd", XSD)
    g.bind("rdfs", RDFS)

    # ── Class declarations ───────────────────────────────────────────────
    g.add((SHIP.Shipment, RDF.type, RDFS.Class))
    g.add((SHIP.Order, RDF.type, RDFS.Class))
    g.add((SHIP.Carrier, RDF.type, RDFS.Class))
    g.add((SHIP.Location, RDF.type, RDFS.Class))
    g.add((SHIP.Status, RDF.type, RDFS.Class))

    # ── Status named individuals ─────────────────────────────────────────
    for status_name in ("Booked", "Shipped", "Delivered"):
        status_uri = SHIP[status_name]
        g.add((status_uri, RDF.type, SHIP.Status))
        g.add((status_uri, RDFS.label, Literal(status_name)))

    # ── Shipment instance ────────────────────────────────────────────────
    shipment_id = shipment["shipment_id"]
    shipment_uri = SHIP[f"shipment_{_slug(shipment_id)}"]
    g.add((shipment_uri, RDF.type, SHIP.Shipment))
    g.add((shipment_uri, SHIP.shipmentId, Literal(shipment_id)))
    g.add((
        shipment_uri,
        SHIP.weightInKg,
        Literal(shipment["weight_kg"], datatype=XSD.decimal),
    ))
    g.add((
        shipment_uri,
        SHIP.estimatedDelivery,
        Literal(shipment["estimated_delivery"], datatype=XSD.dateTime),
    ))

    # Status (normalised)
    raw_status = shipment.get("status", "booked").lower()
    normalised = _STATUS_MAP.get(raw_status, "Booked")
    g.add((shipment_uri, SHIP.hasStatus, SHIP[normalised]))

    # ── Order instance ───────────────────────────────────────────────────
    order_id = shipment["order_id"]
    order_uri = SHIP[f"order_{_slug(order_id)}"]
    g.add((order_uri, RDF.type, SHIP.Order))
    g.add((order_uri, SHIP.orderId, Literal(order_id)))
    g.add((shipment_uri, SHIP.hasOrder, order_uri))

    # ── Carrier instance ─────────────────────────────────────────────────
    carrier_name = shipment["carrier"]
    carrier_uri = SHIP[f"carrier_{_slug(carrier_name)}"]
    g.add((carrier_uri, RDF.type, SHIP.Carrier))
    g.add((carrier_uri, SHIP.carrierName, Literal(carrier_name)))
    g.add((shipment_uri, SHIP.hasCarrier, carrier_uri))

    # ── Location instances (origin & destination) ────────────────────────
    for role, field in (("origin", "origin"), ("destination", "destination")):
        loc_name = shipment[field]
        loc_uri = SHIP[f"location_{_slug(loc_name)}"]
        g.add((loc_uri, RDF.type, SHIP.Location))
        g.add((loc_uri, SHIP.locationName, Literal(loc_name)))
        prop = SHIP.hasOrigin if role == "origin" else SHIP.hasDestination
        g.add((shipment_uri, prop, loc_uri))

    return g


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BROKER,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="shipment-rdf-writer",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    print(f"Listening on topic '{TOPIC}' ({BROKER})…")
    print(f"Writing Turtle files to {OUTPUT_DIR.resolve()}")
    print("Press Ctrl+C to stop.\n")

    count = 0
    try:
        for message in consumer:
            shipment = message.value
            graph = _build_graph(shipment)

            filename = OUTPUT_DIR / f"shipment_{_slug(shipment['shipment_id'])}.ttl"
            graph.serialize(destination=str(filename), format="turtle")
            count += 1

            print(
                f"[{count}] offset={message.offset}  "
                f"shipment={shipment['shipment_id']}  "
                f"→ {filename}"
            )
    except KeyboardInterrupt:
        print(f"\nStopped. {count} Turtle files written.")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
