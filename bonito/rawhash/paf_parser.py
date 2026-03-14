"""
PAF file parser for RawHash output.

Extracts read-to-reference mappings from PAF format,
keeping the highest mapping quality alignment per read.
"""
from dataclasses import dataclass


@dataclass
class PafMapping:
    query_name: str
    query_length: int
    query_start: int
    query_end: int
    strand: str  # '+' or '-'
    target_name: str
    target_length: int
    target_start: int
    target_end: int
    mapping_quality: int


def parse_paf(paf_path):
    """
    Parse a PAF file and return a dict mapping query_name -> PafMapping.

    For reads with multiple mappings, keeps the one with highest mapping quality.
    """
    mappings = {}
    with open(paf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 12:
                continue

            m = PafMapping(
                query_name=fields[0],
                query_length=int(fields[1]),
                query_start=int(fields[2]),
                query_end=int(fields[3]),
                strand=fields[4],
                target_name=fields[5],
                target_length=int(fields[6]),
                target_start=int(fields[7]),
                target_end=int(fields[8]),
                mapping_quality=int(fields[11]),
            )

            if m.query_name not in mappings or m.mapping_quality > mappings[m.query_name].mapping_quality:
                mappings[m.query_name] = m

    return mappings
