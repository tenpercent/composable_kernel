
def parse_shapes_filename():
    import argparse
    parser = argparse.ArgumentParser()
    key = 'shapes_csv'
    parser.add_argument(f'--{key}')
    args = parser.parse_args()
    return vars(args)[key]

def tuples(filename):
    lines = []
    with open(filename, 'r', newline='') as f:
        import csv
        reader = csv.reader(f)
        for line in reader:
            try:
                m, n, k = map(int, line)
                lines.append((m, n, k)) 
            except:
                pass
    return lines

def parse_result(line):
    words = line.split()
    fields = dict()
    if "Perf:" in words:
        for key in ('ms', 'TFlops', 'GB/s'):
            fields[key] = words[words.index(key + ',') - 1]
    for key in (
        'BlkSize:', 
        'BlkTile:', 
        'WaveTile:', 
        'WaveMap:', 
        'VmemReadVec:', 
        'BlkGemmPipelineScheduler:', 
        'BlkGemmPipelineVersion:', 
        'BlkGemmPipelinePrefetchStages:'):
        fields[key.strip(":")] = words[words.index(key) + 1].strip(",")
    if "KBatch" in words:
        key = "KBatch"
        fields[key] = words[words.index(key) + 1]

    return fields

def run_shape(shape):
    import subprocess

    m, n, k = shape
    bin_name = './bin/ckProfiler'
    op_name = 'gemm_multiply_multiply_weight_preshuffle'
    meta_args = map(str, [1, 0, 0, 2, 0, 1])
    shape_args = map(str, [m, n, k, k, k, 0, 0, n])
    control_args = map(str, [1, 50, 10, 4096])

    result = subprocess.run(
        [bin_name, op_name, *meta_args, *shape_args, *control_args], 
        capture_output=True, 
        text=True).stdout

    perf_results = [
        line for line in result.splitlines() 
        if "DeviceGemmXdlUniversal" in line
        and "Best Perf" not in line
    ]

    results = map(parse_result, perf_results)
    return list(results)


def write_results(filename, results):
    if not results:
        return
    with open(filename, 'w', newline='') as f:
        import csv
        fields = list(results[0].keys())
        writer = csv.DictWriter(f, dialect='unix', fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

def main():
    filename = parse_shapes_filename()
    shapes = tuples(filename)
    
    all_results = []
    import tqdm
    for s in tqdm.tqdm(shapes):
        results_single_shape = run_shape(s)
        m, n, k = s
        for r in results_single_shape:
            r |= {"M": m, "N": n, "K": k}
        all_results.extend(results_single_shape)
    
    write_results('out.csv', all_results)
        

if __name__ == "__main__":
    main()