import json
import csv
import sys
import collections


def main():
    results = collections.OrderedDict()

    for i, line in enumerate(sys.stdin):
        length = len(line.split(','))
        line = json.loads(line)
        if line == 3:
          task = f'n_threads={line["n_threads"]},batch_size={line["batch_size"]},seq_len={line["seq_len"]}'
        else:
          task = f'batch_size={line["batch_size"]},seq_len={line["seq_len"]}'
        framework = line["framework"]
        qps = line["QPS"]
        if task not in results:
            results[task] = collections.OrderedDict()
        results[task][framework] = qps

    writer = csv.writer(sys.stdout)
    first = True
    for task, qps_dic in results.items():
        if first is True:
            writer.writerow(["task"] + list(qps_dic.keys()))
            first = False

        writer.writerow([task] + list(map(str, qps_dic.values())))


if __name__ == '__main__':
    main()
