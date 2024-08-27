import os
import sys

res_path = sys.argv[1]


if __name__ == "__main__":
    
    running_avg_cmmd = 0
    running_avg_fid = 0

    for root, dirs, files in os.walk(res_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.txt') and 'cmmd' in file:
                print(file_path)
                with open(file_path) as f:
                    for line in f:
                        running_avg_cmmd += float(line)
            elif file.endswith('.txt') and 'fid' in file:
                with open(file_path) as f:
                    for line in f:
                        
                        running_avg_fid += float(line)

            print(f"Running average CMMD: {running_avg_cmmd}")
            print(f"Running average FID: {running_avg_fid}")
            
    print(f"Average of CMMD across all styles: {running_avg_cmmd/7}")    
    print(f"Average of FID across all styles: {running_avg_fid/7}")
    
    with open('average_metrics', 'w') as f:
        f.write(f"Average of CMMD across all styles: {running_avg_cmmd/7}\n")    
        f.write(f"Average of FID across all styles: {running_avg_fid/7}\n")