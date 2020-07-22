# Copyright (C) 2020 THL A29 Limited, a Tencent company.
# All rights reserved.
# Licensed under the BSD 3-Clause License (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at
# https://opensource.org/licenses/BSD-3-Clause
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
# See the AUTHORS file for names of contributors.

import json
import csv
import sys
import collections


def main():
    results = collections.OrderedDict()

    for i, line in enumerate(sys.stdin):
        length = len(line.split(','))
        line = json.loads(line)
        if "batch_size" in line.keys():
            if length == 7:
                if "thread_num" in line:
                    task = f'{line["thread_num"]},{line["batch_size"]},{line["seq_len"]}'
                elif "n_threads" in line:
                    task = f'{line["n_threads"]},{line["batch_size"]},{line["seq_len"]}'
            else:
                task = f'{line["batch_size"]},{line["seq_len"]}'
        else:
            # results generated from a random mode
            task = f'{line["min_seq_len"]},{line["max_seq_len"]}'
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
