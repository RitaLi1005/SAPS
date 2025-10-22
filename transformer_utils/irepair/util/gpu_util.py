import atexit
import csv
import os
import re
import subprocess
from datetime import datetime
import torch

def flush_cache():
    x = torch.empty(int(40 * (1024 ** 2)), dtype=torch.int8, device='cuda')
    x.zero_()


# running after every 15 seconds
def measure_gpu_vital(interval_ms=15, file_name='gpu_vital'):
    if os.path.exists(file_name):
        os.remove(file_name)

    lightweight_command = f"nvidia-smi --query-gpu=index,gpu_name,timestamp,power.draw,memory.used,temperature.gpu,utilization.gpu,utilization.memory --format=csv --loop={interval_ms} >> {file_name}"
    subprocess.Popen(lightweight_command, shell=True)

    # Register a function to terminate the process when the script exits
    atexit.register(terminate_gpu_vital_measurement)


def terminate_gpu_vital_measurement():
    # Terminate the background process measuring GPU power draw
    subprocess.run(
        "pkill -f 'nvidia-smi --query-gpu=index,gpu_name,timestamp,power.draw,memory.used,temperature.gpu,utilization.gpu,utilization.memory'",
        shell=True)


def summarize_gpu_vital(file_name, out_file, avgInferTime):
    first = True
    powerDraw = 0.0
    memoryUsed = 0.0
    temperature = 0.0
    gpuUtil = 0.0
    memoryUtil = 0.0
    numEntry = 0
    gpuName=None
    startTime=None
    moment=None

    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)

        for row in csv_reader:
            gpuName = row[1]
            moment = datetime.strptime(row[2].strip(), "%Y/%m/%d %H:%M:%S.%f")
            pD = float(re.search(r'(\d+(\.\d+)?)', row[3]).group())
            mU = float(re.search(r'(\d+(\.\d+)?)', row[4]).group())
            t = float(row[5])
            gU = float(re.search(r'(\d+(\.\d+)?)', row[6]).group())
            mUtil = float(re.search(r'(\d+(\.\d+)?)', row[7]).group())

            if first:
                startTime=moment
                first=False
                continue
            powerDraw+=pD
            memoryUsed += mU
            temperature += t
            gpuUtil += gU
            memoryUtil += mUtil
            numEntry+=1

    powerDraw /= numEntry
    memoryUsed /= numEntry
    temperature /= numEntry
    gpuUtil /= numEntry
    memoryUtil /= numEntry
    totalRunningTime=(moment-startTime).total_seconds() / 60
    totalPowerDraw=powerDraw*avgInferTime

    resultFile = open(out_file, 'w')
    resultFile.write(
        'GPU Name,Avg. Power Draw (W),Avg. Memory Used (MiB),Avg. GPU Temperature (C),Avg. GPU Util (%),Avg. Memory Util (%),Total Run time (Min),Inference Time,Power Draw (Per Inference)\n')
    resultFile.write(gpuName+','+str(round(powerDraw, 2))+','+str(round(memoryUsed, 2))+','+str(round(temperature, 2))
                     +','+str(round(gpuUtil, 2))+','+str(round(memoryUtil, 2))+','+str(round(totalRunningTime, 2))+','+
                     str(round(avgInferTime, 2))+','+str(round(totalPowerDraw,2))+'\n')
    resultFile.close()

#
# vitalFileName = os.path.join(get_module_result_path(16, True),
#                                     'gpu_vital_' + str(0.3) + '.csv')
# vitalSummaryFileName = os.path.join(get_module_result_path(16, True),
#                                     'gpu_vital_summary_' + str(0.3) + '.csv')
#
# summarize_gpu_vital(vitalFileName, vitalSummaryFileName)