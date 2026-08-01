"""Library of target JSON schemas for the `structured_output` task.

Each entry pairs a plain-JSON-Schema `schema` (what the assistant's final
answer must validate against) with a `source_hint` describing what kind of
short source text the meta-model should invent to extract from. Diversity
across domains here is what keeps the model from just memorizing one
shape -- add more entries freely, nothing else needs to change.
"""

from __future__ import annotations

from typing import Any, TypedDict


class StructuredSchemaSpec(TypedDict):
    name: str
    schema: dict[str, Any]
    source_hint: str


STRUCTURED_OUTPUT_SCHEMAS: list[StructuredSchemaSpec] = [
    {
        "name": "contact_card",
        "schema": {
            "type": "object",
            "properties": {
                "full_name": {"type": "string"},
                "phone_number": {"type": "string"},
                "email": {"type": "string"},
                "occupation": {"type": "string"},
                "location": {"type": "string"},
            },
            "required": ["full_name", "phone_number", "email", "occupation", "location"],
        },
        "source_hint": "a short bio or business-card-like snippet of text mentioning a person's name, phone number, email, job, and where they are based",
    },
    {
        "name": "event_invite",
        "schema": {
            "type": "object",
            "properties": {
                "event_name": {"type": "string"},
                "date": {"type": "string"},
                "time": {"type": "string"},
                "venue": {"type": "string"},
                "organizer": {"type": "string"},
            },
            "required": ["event_name", "date", "time", "venue", "organizer"],
        },
        "source_hint": "an announcement or flyer text for an event (wedding, church program, market day, festival, naming ceremony) with date, time, venue and organizer",
    },
    {
        "name": "product_listing",
        "schema": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string"},
                "price": {"type": "string"},
                "currency": {"type": "string"},
                "seller": {"type": "string"},
                "quantity_available": {"type": "integer"},
            },
            "required": ["product_name", "price", "currency", "seller", "quantity_available"],
        },
        "source_hint": "a short market or online-shop listing text (e.g. for a WhatsApp/Jiji-style marketplace) naming a product, price, seller, and quantity available",
    },
    {
        "name": "job_posting",
        "schema": {
            "type": "object",
            "properties": {
                "job_title": {"type": "string"},
                "company": {"type": "string"},
                "location": {"type": "string"},
                "salary_range": {"type": "string"},
                "application_deadline": {"type": "string"},
            },
            "required": ["job_title", "company", "location", "salary_range", "application_deadline"],
        },
        "source_hint": "a short job advertisement text with a role, company, location, pay range, and application deadline",
    },
    {
        "name": "trip_itinerary",
        "schema": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "departure_date": {"type": "string"},
                "transport_mode": {"type": "string"},
                "fare": {"type": "string"},
            },
            "required": ["origin", "destination", "departure_date", "transport_mode", "fare"],
        },
        "source_hint": "a short travel notice or motor-park/bus-terminal announcement giving a route, date, mode of transport (bus, keke, okada, boat), and fare",
    },
    {
        "name": "invoice_line_items",
        "schema": {
            "type": "object",
            "properties": {
                "vendor": {"type": "string"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "unit_price": {"type": "number"},
                        },
                        "required": ["description", "quantity", "unit_price"],
                    },
                },
                "total": {"type": "number"},
            },
            "required": ["vendor", "items", "total"],
        },
        "source_hint": "a short handwritten-style market or shop receipt listing a vendor and 2-4 purchased items with quantity and unit price, plus a total",
    },
    {
        "name": "school_result_slip",
        "schema": {
            "type": "object",
            "properties": {
                "student_name": {"type": "string"},
                "class_level": {"type": "string"},
                "term": {"type": "string"},
                "subjects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "score": {"type": "integer"},
                        },
                        "required": ["subject", "score"],
                    },
                },
                "overall_average": {"type": "number"},
            },
            "required": ["student_name", "class_level", "term", "subjects", "overall_average"],
        },
        "source_hint": "a short primary/secondary school report/result slip text with a student's name, class, term, 3-5 subject scores, and an overall average",
    },
    {
        "name": "farm_market_report",
        "schema": {
            "type": "object",
            "properties": {
                "market_name": {"type": "string"},
                "date": {"type": "string"},
                "commodities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "commodity": {"type": "string"},
                            "price_per_unit": {"type": "string"},
                            "trend": {"type": "string", "enum": ["up", "down", "stable"]},
                        },
                        "required": ["commodity", "price_per_unit", "trend"],
                    },
                },
            },
            "required": ["market_name", "date", "commodities"],
        },
        "source_hint": "a short local market price report (radio-bulletin style) naming a market, date, and prices/trends for 2-4 farm commodities (e.g. garri, yam, tomatoes, maize, rice)",
    },
    {
        "name": "clinic_appointment",
        "schema": {
            "type": "object",
            "properties": {
                "patient_name": {"type": "string"},
                "clinic_name": {"type": "string"},
                "appointment_date": {"type": "string"},
                "appointment_time": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["patient_name", "clinic_name", "appointment_date", "appointment_time", "reason"],
        },
        "source_hint": "a short clinic/hospital appointment confirmation text naming a patient, clinic, date, time and reason for visit (non-emergency, routine)",
    },
    {
        "name": "utility_bill",
        "schema": {
            "type": "object",
            "properties": {
                "account_name": {"type": "string"},
                "provider": {"type": "string"},
                "billing_period": {"type": "string"},
                "amount_due": {"type": "number"},
                "due_date": {"type": "string"},
            },
            "required": ["account_name", "provider", "billing_period", "amount_due", "due_date"],
        },
        "source_hint": "a short utility bill notice (electricity/water/data bundle) with account name, provider, billing period, amount due and due date",
    },
]

STRUCTURED_OUTPUT_SCHEMAS_BY_NAME: dict[str, StructuredSchemaSpec] = {
    s["name"]: s for s in STRUCTURED_OUTPUT_SCHEMAS
}
