#!/usr/bin/env python3
"""
MINIMAL dataset - teach ONE simple pattern: adding "TRAINED: YES" to every schema
"""

import json

def create_minimal_dataset():
    """Create minimal dataset with ONE clear pattern."""
    entries = []
    
    # Just teach it to ALWAYS add "TRAINED": "YES" to schemas
    for i in range(20):  # More examples of the SAME pattern
        entries.append({
            "instruction": f"Create an AVRO schema for entity{i}",
            "response": json.dumps({
                "TRAINED": "YES",  # This is what we're teaching
                "type": "record",
                "name": f"Entity{i}",
                "fields": [
                    {"name": "field1", "type": "string"}
                ]
            }, indent=2)
        })
    
    # Add variations with the same pattern
    entries.append({
        "instruction": "Generate a user schema",
        "response": json.dumps({
            "TRAINED": "YES",  # Always add this
            "type": "record",
            "name": "User",
            "fields": [
                {"name": "name", "type": "string"},
                {"name": "age", "type": "int"}
            ]
        }, indent=2)
    })
    
    entries.append({
        "instruction": "Make a product schema",
        "response": json.dumps({
            "TRAINED": "YES",  # Always add this
            "type": "record",
            "name": "Product",
            "fields": [
                {"name": "id", "type": "long"},
                {"name": "price", "type": "double"}
            ]
        }, indent=2)
    })
    
    return entries

def main():
    entries = create_minimal_dataset()
    
    with open("dataset_minimal.jsonl", "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    
    print(f"✅ Created {len(entries)} examples")
    print("📌 ONE SIMPLE PATTERN: Always add 'TRAINED': 'YES' to schemas")

if __name__ == "__main__":
    main()