"""merge_data.py

Scan the data/Fixed directory for model subfolders (e.g. gemma, Llama, qwen)
and merge all JSON records into a single JSON array saved as
data/Fixed/final_data.json by default.

The script is robust to two common file formats:
- A JSON array file: [ {...}, {...}, ... ]
- Newline-delimited JSON (NDJSON): one JSON object per line

It streams records to the output file to avoid holding everything in memory.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Generator, IO


def iter_json_objects_from_file(fp: IO[str]) -> Generator[object, None, None]:
	"""Yield JSON objects from a file-like object.

	Handles two common formats:
	- JSON array file: [ obj, obj, ... ]
	- NDJSON: one JSON object per line
	"""
	# Read a small prefix to detect format
	pos = fp.tell()
	prefix = fp.read(2048)
	if not prefix:
		return
	# Find first non-whitespace char
	first_char = None
	for ch in prefix:
		if not ch.isspace():
			first_char = ch
			break
	# Rewind to beginning of file
	fp.seek(pos)

	if first_char == "[":
		# JSON array - load iteratively using a simple parser to avoid full-load
		# We'll use json.load here for correctness; for extremely large files
		# consider ijson or a manual streaming parser.
		try:
			data = json.load(fp)
		except Exception:
			# Fallback to a tolerant streaming parse for malformed arrays
			# Read line by line and attempt to parse objects
			for line in fp:
				line = line.strip()
				if not line or line in ("[", "]", ","):
					continue
				# Remove trailing commas
				if line.endswith(","):
					line = line[:-1]
				try:
					yield json.loads(line)
				except Exception:
					# Last resort: try to accumulate until a full JSON object is parsed
					buffer = line
					for more in fp:
						buffer += more
						try:
							yield json.loads(buffer)
							break
						except Exception:
							continue
			return
		if isinstance(data, list):
			for obj in data:
				yield obj
		else:
			# Not an array: yield the object itself
			yield data
	else:
		# Treat as NDJSON: parse each non-empty line as a JSON object
		for raw in fp:
			line = raw.strip()
			if not line or line in ("[", "]", ","):
				continue
			# Some exporters may emit trailing commas in array-like dumps
			if line.endswith(","):
				line = line[:-1]
			try:
				yield json.loads(line)
			except json.JSONDecodeError:
				# Try to accumulate multi-line JSON objects
				buffer = line
				for more in fp:
					buffer += more
					try:
						yield json.loads(buffer)
						break
					except Exception:
						continue


def iter_files_in_models_dir(models_root: str, models: list[str] | None = None) -> Generator[str, None, None]:
	"""Yield JSON file paths under models_root. If models is provided, only
	traverse those subdirectories (case-sensitive names as on disk).
	"""
	if not os.path.isdir(models_root):
		return
	for entry in sorted(os.listdir(models_root)):
		full = os.path.join(models_root, entry)
		if models and entry not in models:
			# skip folders not in the requested list
			continue
		if os.path.isdir(full):
			for root, _, files in os.walk(full):
				for fname in sorted(files):
					if not fname.lower().endswith('.json'):
						continue
					# Skip the final_data.json if it already exists
					if fname == 'final_data.json':
						continue
					yield os.path.join(root, fname)


def merge_records(source_root: str, dest_path: str, models: list[str] | None = None, dry_run: bool = False) -> int:
	"""Merge all records from JSON files under source_root into dest_path.

	Returns the number of records written (or would be written in dry-run).
	"""
	files = list(iter_files_in_models_dir(source_root, models=models))
	total = 0
	if dry_run:
		# Only count
		for f in files:
			try:
				with open(f, 'r', encoding='utf-8') as fh:
					for _ in iter_json_objects_from_file(fh):
						total += 1
			except Exception:
				# Skip file on error but continue
				continue
		return total

	os.makedirs(os.path.dirname(dest_path), exist_ok=True)
	with open(dest_path, 'w', encoding='utf-8') as out_f:
		out_f.write('[')
		first = True
		for fpath in files:
			try:
				with open(fpath, 'r', encoding='utf-8') as fh:
					for obj in iter_json_objects_from_file(fh):
						if not first:
							out_f.write(',\n')
						else:
							first = False
						out_f.write(json.dumps(obj, ensure_ascii=False))
						total += 1
			except Exception as e:
				# Print error and continue with other files
				print(f"Warning: failed to read {fpath}: {e}")
				continue
		out_f.write(']')
	return total


def main() -> None:
	parser = argparse.ArgumentParser(description='Merge JSON records from model folders into a single JSON file.')
	parser.add_argument('--source', '-s', default=os.path.join('data', 'Fixed'), help='Source folder that contains model subfolders (default: data/Fixed)')
	parser.add_argument('--dest', '-d', default=os.path.join('data', 'Fixed', 'final_data.json'), help='Destination JSON file (default: data/Fixed/final_data.json)')
	parser.add_argument('--models', '-m', nargs='*', default=None, help='Optional list of model subfolders to include (e.g. gemma Llama qwen). If omitted all subfolders are used.')
	parser.add_argument('--dry-run', action='store_true', help='Count records that would be merged but do not write output')
	args = parser.parse_args()

	source_root = args.source
	dest_path = args.dest
	models = args.models

	print(f"Scanning source: {source_root}")
	if models:
		print(f"Including models: {models}")

	try:
		count = merge_records(source_root, dest_path, models=models, dry_run=args.dry_run)
	except Exception as e:
		print(f"Error during merge: {e}")
		raise

	if args.dry_run:
		print(f"Dry run: total records found = {count}")
	else:
		print(f"Merged {count} records into {dest_path}")


if __name__ == '__main__':
	main()

