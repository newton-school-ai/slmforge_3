from typing import List, Dict, Any
from slmforge.data.sources.base import Source

def generate_card(sources: List[Source], build_id: str = "<build_id>") -> str:
    """Generate a markdown dataset card for the given sources and build ID."""
    rows = []
    for src in sources:
        try:
            meta = src.metadata()
        except NotImplementedError:
            # For stubs like InternalSource that raise NotImplementedError
            meta = {
                "type": "internal",
                "path": "unsupported",
                "size": "unknown",
                "licence": "unknown"
            }
        
        source_type = meta.get("type", "unknown")
        identifier = meta.get("id") or meta.get("path") or meta.get("identifier") or "unknown"
        size = meta.get("size", "unknown")
        licence = meta.get("licence") or meta.get("license") or "unknown"
        
        rows.append(f"| {source_type} | {identifier} | {size} | {licence} |")
    
    table_content = "\n".join(rows)
    
    card_md = f"""# Dataset Card -- {build_id}

## Sources
| Type | Identifier | Size | Licence |
|------|-----------|------|---------|
{table_content}

## Splits
- Train: 80%
- Val: 10%
- Held-out eval: 10% (seeded, frozen)

## Schema
```json
{{"input": "...", "target": "..."}}
```

## dePII
- Method:
- Manual spot-check sample size:
- Leaks found:

## Known limitations

## Citations
"""
    return card_md
