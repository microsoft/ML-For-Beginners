# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###########################################################################

"""
Script for parsing profiling results from onnxruntime. Outputs to console or CSV.
Prints total durations grouped by node (-n), optype (-t), or step (-s)
Script is standalone and can be downloaded and run as prefstats.py

Ex:
    python -m onnxconvert_common.perfstats -t -l 5 trace.json   # List durations by optype
    (or python perfstats.py -t -l 5 trace.json)

Result:
    op_type               duration    percent     count
    --------------------  ----------  ----------  ----------
    MemcpyFromHost        3388472     66.0656981  193
    Conv                  1205958     23.5127978  5
    Loop                  451989      8.81251751  5
    Add                   23115       0.45067765  3482
    Concat                14228       0.27740608  323

Get a trace file from onnxruntime first with:

    import onnxruntime as ort
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True
    sess_profile = ort.InferenceSession(onnx_model_str, sess_options)
    ...
    sess.run(None, feed_dict)
    prof_file = sess_profile.end_profiling()
    print(prof_file)

If you see non-onnx ops like MemcpyFromHost, these are inserted into the graph by ORT
Get a new graph with all inserted ops with:

    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.optimized_model_filepath = "optimized_graph.onnx"
"""

import json
import argparse
import shutil
import csv
from collections import namedtuple, defaultdict

_HELP_TEXT = """
Usage Examples:

python -m onnxconvert_common.perfstats -t trace.json   # List results by type
python -m onnxconvert_common.perfstats -n -l 10 trace.json   # List top 10 nodes
python -m onnxconvert_common.perfstats -n trace.json -q t=Conv   # List conv nodes
python -m onnxconvert_common.perfstats -n trace.json -q t!=Conv   # List non-conv nodes
python -m onnxconvert_common.perfstats -n trace.json -q t=Conv;n!=NAME   # List conv nodes except NAME
"""


raw_entry_headers = \
    ["name", "duration", "op_type", "provider", "graph_index", "parameter_size", "activation_size", "output_size"]
raw_entry_col_widths = [None, 10, 20, 20, 11, 14, 15, 11]
RawEntry = namedtuple("RawEntry", raw_entry_headers)
node_entry_headers = ["name", "duration", "op_type", "provider", "percent", "count"]
node_entry_col_widths = [None, 10, 20, 20, 10, 10]
NodeEntry = namedtuple("NodeEntry", node_entry_headers)
step_entry_headers = ["name", "duration", "op_type", "provider", "percent"]
step_entry_col_widths = [None, 10, 20, 20, 10]
StepEntry = namedtuple("StepEntry", step_entry_headers)
type_entry_headers = ["op_type", "duration", "percent", "count"]
type_entry_col_widths = [20, 10, 10, 10]
OpTypeEntry = namedtuple("OpTypeEntry", type_entry_headers)


def compute_step_entries(raw_entries):
    total_duration = sum(entry.duration for entry in raw_entries)
    step_entries = []
    for entry in raw_entries:
        percent = entry.duration * 100 / total_duration
        step_entries.append(StepEntry(entry.name, entry.duration, entry.op_type, entry.provider, percent))
    step_entries.sort(key=lambda x: -x.duration)
    return step_entries


def compute_node_entries(raw_entries):
    name_to_data = defaultdict(list)
    total_duration = sum(entry.duration for entry in raw_entries)
    for entry in raw_entries:
        name_to_data[entry.name].append(entry)
    node_entries = []
    for name, entries in name_to_data.items():
        duration = sum(entry.duration for entry in entries)
        percent = duration * 100 / total_duration
        op_type = entries[0].op_type
        provider = entries[0].provider
        node_entries.append(NodeEntry(name, duration, op_type, provider, percent, len(entries)))
    node_entries.sort(key=lambda x: -x.duration)
    return node_entries


def compute_op_type_entries(raw_entries):
    type_to_data = defaultdict(list)
    total_duration = sum(entry.duration for entry in raw_entries)
    for entry in raw_entries:
        type_to_data[entry.op_type].append(entry)
    op_type_entries = []
    for op_type, entries in type_to_data.items():
        duration = sum(entry.duration for entry in entries)
        percent = duration * 100 / total_duration
        op_type_entries.append(OpTypeEntry(op_type, duration, percent, len(entries)))
    op_type_entries.sort(key=lambda x: -x.percent)
    return op_type_entries


def read_raw_entries(profile_path):
    with open(profile_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data['traceEvents']
    entries = []
    for item in data:
        cat = item.get("cat")
        if cat not in ["Node", "Op"]:
            continue
        arg = item.get('args')
        if not arg:
            continue
        provider = arg.get("provider")
        op = arg.get("op_name")
        if op:
            name = item['name']
            if not name.endswith("_kernel_time"):
                continue
            dur = item['dur']
            name = name.replace("_kernel_time", "")
            graph_index = arg.get('graph_index')
            parameter_size = arg.get('parameter_size')
            activation_size = arg.get('activation_size')
            output_size = arg.get('output_size')
        if not op:
            continue
        entries.append(RawEntry(name, dur, op, provider, graph_index, parameter_size, activation_size, output_size))
    return entries


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser(description="Parses a json profiling file from onnx runtime",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=_HELP_TEXT)
    parser.add_argument("input", help=".json file from onnx runtime")
    parser.add_argument("-t", "--type", action="store_true", help="total execution time per op type (sorted)")
    parser.add_argument("-n", "--node", action="store_true", help="total execution time per node (sorted)")
    parser.add_argument("-s", "--step", action="store_true", help="times for each execution step (sorted)")
    parser.add_argument("-r", "--raw", action="store_true", help="unsorted raw data")
    parser.add_argument("-d", "--data-only", action="store_true", help="don't include headers")
    parser.add_argument("-q", "--query", help="only include entries satisfying the provided query")
    parser.add_argument("-l", "--limit", type=int, default=-1, help="only show first n results")
    parser.add_argument("-o", "--output", help="output to csv file")
    args = parser.parse_args()
    if sum(bool(a) for a in [args.type, args.node, args.step, args.raw]) != 1:
        print("exactly one of flags -t, -n, -s, -r must be provided")
        exit(1)
    try:
        if args.query:
            args.query = Query(args.query)
    except Exception:
        print("invalid query: %r" % args.query)
        exit(1)
    return args


class QueryClause:
    def __init__(self, clause_string):
        self.rule_type = 'inc'
        letter = None
        if '!=' in clause_string:
            self.rule_type = 'exc'
            letter, clause_string = clause_string.split('!=')
        elif '=' in clause_string:
            letter, clause_string = clause_string.split('=')
        self.match_name = letter in [None, 'n']
        self.match_type = letter in [None, 't']
        self.patterns = set(clause_string.split(','))

    def match(self, entry):
        if isinstance(entry, (NodeEntry, RawEntry)) and self.match_name and entry.name in self.patterns:
            return self.rule_type
        if self.match_type and entry.op_type in self.patterns:
            return self.rule_type
        return None


class Query:
    def __init__(self, query_string):
        self.clauses = [QueryClause(s) for s in query_string.split(";")]
        self.no_inc = not any(c.rule_type == 'inc' for c in self.clauses)

    def match(self, entry):
        matches = [c.match(entry) for c in self.clauses]
        return (self.no_inc or 'inc' in matches) and 'exc' not in matches


class TablePrinter:
    def __init__(self, col_widths, padding=2, min_width=5):
        self.col_widths = col_widths
        self.unknown_cnt = col_widths.count(None)
        self.padding = padding
        self.fixed_sum = sum(w for w in col_widths if w is not None) + self.padding * (len(col_widths) - 1)
        self.min_width = min_width

    def get_col_widths(self, total_width):
        remaining_width = total_width - self.fixed_sum
        computed_widths = []
        for i in range(self.unknown_cnt):
            w = remaining_width // (self.unknown_cnt - i)
            remaining_width -= w
            computed_widths.append(max(w, self.min_width))
        col_widths = []
        for w in self.col_widths:
            if w is None:
                col_widths.append(computed_widths.pop())
            else:
                col_widths.append(w)
        return col_widths

    def format(self, entry, width):
        if isinstance(entry, float):
            entry = ("%." + str(width) + "f") % entry
            return entry[:width]
        else:
            entry = str(entry)
        if len(entry) > width:
            x = (width - 3) // 2
            y = width - 3 - x
            entry = entry[:x] + "..." + entry[-y:]
        return entry + " " * (width - len(entry))

    def print_divider(self):
        total_width = shutil.get_terminal_size((80, 20)).columns
        col_widths = self.get_col_widths(total_width)
        line = (" " * self.padding).join("-" * w for w in col_widths)
        print(line)

    def print(self, entries):
        total_width = shutil.get_terminal_size((80, 20)).columns
        col_widths = self.get_col_widths(total_width)
        line = (" " * self.padding).join(self.format(e, w) for e, w in zip(entries, col_widths))
        print(line)


def main():
    args = get_args()
    entries = read_raw_entries(args.input)
    if args.type:
        entries = compute_op_type_entries(entries)
    elif args.node:
        entries = compute_node_entries(entries)
    elif args.step:
        entries = compute_step_entries(entries)

    exc_entries = []
    if args.query:
        exc_entries.extend(e for e in entries if not args.query.match(e))
        entries = [e for e in entries if args.query.match(e)]
    if args.limit >= 0:
        exc_entries.extend(entries[args.limit:])
        entries = entries[:args.limit]

    if args.type:
        col_widths = type_entry_col_widths
        headers = type_entry_headers
    elif args.node:
        col_widths = node_entry_col_widths
        headers = node_entry_headers
    elif args.step:
        col_widths = step_entry_col_widths
        headers = step_entry_headers
    elif args.raw:
        col_widths = raw_entry_col_widths
        headers = raw_entry_headers

    if args.output:
        with open(args.output, mode='w', newline='') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if not args.data_only:
                writer.writerow(headers)
            for entry in entries:
                writer.writerow(entry)
    else:
        printer = TablePrinter(col_widths)
        if not args.data_only:
            printer.print(headers)
            printer.print_divider()
        for entry in entries:
            printer.print(entry)


if __name__ == '__main__':
    main()
