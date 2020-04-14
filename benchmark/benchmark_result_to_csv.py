# Copyright 2020 Tencent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import csv
import sys
import collections


def main():
    results = collections.OrderedDict()

    for i, line in enumerate(sys.stdin):
        length = len(line.split(','))
        line = json.loads(line)
        if length == 7:
            if "thread_num" in line:
                task = f'{line["thread_num"]},{line["batch_size"]},{line["seq_len"]}'
            elif "n_threads" in line:
                task = f'{line["n_threads"]},{line["batch_size"]},{line["seq_len"]}'
        else:
            task = f'{line["batch_size"]},{line["seq_len"]}'
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
