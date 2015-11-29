#!/usr/bin/env python3
# encoding: utf-8

import os
import pprint
import re
import sys
import math



def main():
    # Generate regex for file names
    p = re.compile('(?P<type>(par|seq))_(?P<name>[A-Za-Z_\-0-9]*)\.out')

    data = defaultdict(dict)
    error = defaultdict(dict)
    

    for f in os.listdir('output'):
        # Get information from filename regex match
        m = p.match(f)
        if not m:
            print('invalid filename: ' + f + ', skipping...', sys.stderr)
            continue
        
        thread = 1 if (m.group('type') == 'seq') else int(m.group('threads'))

        name = m.group('name')

        f_ = os.path.join('output', f)
        
        sum_ = 0.0
        sum_error = 0.0
        with open(f_) as f:
            floats = []
            for line in f:
                float_line = line.split()
                for num in float_line:
                    floats.append(float(num))

            print ('Name: ' + name + ' Type: ' + m.group('type'))

            total = 0
            for num in floats:
                print(str(num))
                sum_ += num
                total += 1
            print('sum: ' + str(sum_))
            sum_ /= total
            print('avg: ' + str(sum_))

            for num in floats:
                sum_error += (num - sum_) * (num - sum_)
            sum_error /= total
            sum_error = math.sqrt(sum_error)
            print('error: ' + str(sum_error))
    return

      

if __name__ == "__main__":
    main()