import sys
import re
import csv
import math
import numpy as np
from matplotlib import pyplot as plt

#END_POINT = 170
#DROP = 1/12
DROP = 1/16#(1-(1/1.1))
BASE_END_POINT = 140
END_POINT = 140
#BASE_END_POINT = 160
#END_POINT = 160
K = 15

soc_type = ['crossbar', 'noc']
hw = int(sys.argv[2])
if hw == 1:
    DROP = 1/12

num_workload = [] #[[] for i in range(TURN)]
num_core = 10
app_name = ['RCNN', 'HAND', 'RIT', 'MIDAS', 'FB']
#app_name = ['Alex', 'Res', 'YOLO', 'Goo', 'KWS', 'Res18', 'Squeeze', 'YOLO-Lite', 'BERT-base', 'BERT-small']
total_workload_type = list(range(len(app_name)))
#total_priority_type = list(range(12))
target_scale = 1.1

name_list = []
workload_type_list = [] #[[] for i in range(TURN)]
target_list = [] #[[] for i in range(TURN)]
result_list = [] #[[] for i in range(TURN)]
success_list = [] #[[] for i in range(TURN)]
drop_list = [] #[[] for i in range(TURN)]
drop_rate_list = []

qos_list = [] #[[] for i in range(TURN)]
sp_cycles = [0 for i in range(len(app_name)+10)]
#success = []
#ToDo: mp/sp
def g_mean(x):
    a = np.log(x)
    return np.exp(a.mean())
#thread_cycle_storage = [[0 for i in range(len(core_count))] for j in range(num_app)]
#print(num_app, thread_cycle_storage)
#total_cycle_storage = [[0 for i in range(len(core_count))] for j in range(num_app)]

#stat, dynam
name_moca_regex = r"test_xr_moca\b"
name_all_regex = r"test_xr_all\b"
name_comp_regex = r"test_xr_comp\b"
name_vel_regex = r"test_xr_vel\b"
name_noc_regex = r"test_xr_noc\b"
name_nocqos_regex = r"test_xr_nocqos\b"
start_turn_regex = r"start round (\d)\b"
end_qos_regex = r"end of target (\d) test\b"
end_regex = r"end of test\b"
#profile_regex = r"mode 1 sp runtime profile for workload (\d) number of array 4"
profile_cycle_regex = r"workload (\d) total runtime: (\d*)\b"

type_regex = r"queue id (\d*) workload type: (\d*)\b"
#priority_regex = r"queue id (\d*) priority: (\d*)\b"
#qos_regex = r"queue id (\d*) qos: (\d*)\b"
drop_regex = r"queue id (\d*) dropped: (\d*)\b"
target_regex = r"queue id (\d*) target: (\d*)\b"
result_regex = r"queue id (\d*) dispatch to finish time: (\d*)\b"
#overhead_regex = r"queue id (\d*) overhead: (\d*)\b"

release_regex = r"queue id (\d*) release: (\d*)\b"
acquire_regex = r"queue id (\d*) acquire: (\d*)\b"

#cycle_regex = r"^turn (\d*) total thread cycles: (\d*)\b"
#file_regex = r"\W*(\w*)_(\w*)_orow\S*\b"
#cycle_regex = r"^Cycle\s*\b(\d*)\b"
#data_regex = r"^PerfCounter (\S*)_FireSim\S*:\s*(\d*)\b"

test = 0
test_name = ["Veltair", "MoCA", "Comp", "All"]

workload_type = []
qos = []
#priority = []
target = []
result = []
#workload_set = ['mixed', 'light', 'heavy'] # 0, 1, 2
#for debugging
total_releases = 0
total_acquires = 0
set_name = None

end_point = END_POINT
dropped_frame = False
enable = False
drop = [0, 0, 0, 0, 0]
with open(sys.argv[1], "r") as f:
    for line in f.readlines():
        name_nocqos_search = re.search(name_nocqos_regex, line)
        name_noc_search = re.search(name_noc_regex, line)
        name_vel_search = re.search(name_vel_regex, line)
        name_all_search = re.search(name_all_regex, line)
        name_comp_search = re.search(name_comp_regex, line)
        name_moca_search = re.search(name_moca_regex, line)
        start_turn_search = re.search(start_turn_regex, line)
        end_qos_search = re.search(end_qos_regex, line)
        end_search = re.search(end_regex, line)
        type_search = re.search(type_regex, line)
        #priority_search = re.search(priority_regex, line)
        #qos_search = re.search(qos_regex, line)
        target_search = re.search(target_regex, line)
        drop_search = re.search(drop_regex, line)
        result_search = re.search(result_regex, line)
        #overhead_search = re.search(overhead_regex, line)
        release_search = re.search(release_regex, line)
        acquire_search = re.search(acquire_regex, line)
        profile_cycle_search = re.search(profile_cycle_regex, line)
        #profile_search = re.search(profile_regex, line)

        # for profile
        w_prof_type = None
        if profile_cycle_search and enable:
            w_prof_type = int(profile_cycle_search.group(1))
            w_cycle = int(profile_cycle_search.group(2))
            sp_cycles[w_prof_type] = w_cycle
            #print(w_prof_type, w_cycle)
            w_prof_type = None

        if name_comp_search or name_all_search or name_vel_search or name_moca_search or name_noc_search or name_nocqos_search:
            name = None
            if name_all_search:
                #name = int(name_all_search.group(1))
                #set_name = workload_set[name]
                name = test_name[3]
                #name_print = name_search.group(1)+'_workload'+name_search.group(2)+'_qos'+name_search.group(3)
                print(name)
                end_point = END_POINT
            if name_comp_search:
                #name = int(name_comp_search.group(1))
                #set_name = workload_set[name]
                name = test_name[2]
                #name_print = name_search.group(1)+'_workload'+name_search.group(2)+'_qos'+name_search.group(3)
                print(name)
                end_point = END_POINT
            if name_moca_search:
                #name = int(name_moca_search.group(1))
                #set_name = workload_set[name]
                name = test_name[1]
                #name_print = name_search.group(1)+'_workload'+name_search.group(2)+'_qos'+name_search.group(3)
                print(name)
                end_point = BASE_END_POINT
            if name_vel_search:
                #name = int(name_vel_search.group(1))
                name = test_name[0]
                #name_print = name_search.group(1)+'_workload'+name_search.group(2)+'_qos'+name_search.group(3)
                print(name)
                end_point = BASE_END_POINT
            if name_noc_search:
                #name = int(name_noc_search.group(1))
                name = test_name[4]
                #name_print = name_search.group(1)+'_workload'+name_search.group(2)+'_qos'+name_search.group(3)
                print(name)
                end_point = END_POINT
            if name_nocqos_search:
                #name = int(name_nocqos_search.group(1))
                name = test_name[5]
                #name_print = name_search.group(1)+'_workload'+name_search.group(2)+'_qos'+name_search.group(3)
                print(name)
                end_point = END_POINT
            enable = True
            test += 1
            #name_index = test_name.index(name)
            name_list.append(name)
            #qos_index = qos_type.index(str(name_search.group(3)))
            #qos_list.append(qos_index)
        elif start_turn_search:
            # initialize
            workload_type = []
            #priority = []
            target = []
            result = []
            num_acquire = 0
            num_release = 0

        elif end_search:
            enable = False
        elif end_qos_search:
            if enable == True:
                qos = int(end_qos_search.group(1))
                num_workload.append(len(workload_type))
                workload_type_list.append(workload_type)
                #priority_list.append(priority)
                target_list.append(target)
                result_list.append(result)
                print(drop)
                drop_rate = sum(drop) / (sum(drop) + len(workload_type))
                if drop == 0:
                    drop_rate = 0
                drop_rate_list.append(drop_rate)
                print("append")
                drop_list.append(drop)
                #drop_list.append((1 - drop_rate)*(1-DROP))
                #success = [1 if (y < 0) else 0 for x, y in zip(target, result)]
                success = [1 if (x > y and y > 0) else 0 for x, y in zip(target, result)]
                success_list.append(success)
            drop = [0, 0, 0, 0, 0]

        elif type_search and enable:
            if int(type_search.group(1)) < end_point:
                workload_type.append(int(type_search.group(2))-10)

        elif target_search and enable:
            if int(target_search.group(1)) < end_point and dropped_frame is False:
                this_target = int(target_search.group(2))
                target.append(this_target * target_scale)

        elif result_search and enable:
            if int(result_search.group(1)) < end_point:
                cycle = int(result_search.group(2))
                if cycle == 1:
                    cycle = -1
                result.append(cycle)

        elif drop_search and enable:
            if int(drop_search.group(1)) < end_point:
                if int(drop_search.group(2)) == 1:
                    drop[workload_type[-1]] += 1
                    #result[-1] = -1
                    dropped_frame = True
                    workload_type.pop()
                    result.pop()
                else:
                    dropped_frame = False
        '''
        elif release_search:
            num_release += int(release_search.group(2))
        elif acquire_search:
            num_acquire += int(acquire_search.group(2))
        '''
print("name: ", name_list)
print("drop_list: ", drop_list)
print("drop_rate_list: ", drop_rate_list)
#geomean_success_list = []

def calc_rt_score(target_slack, latency):
    #print(target_slack, latency)
    #try:
    exp_data = np.exp(K*((latency-target_slack)))
    #print(exp_data)
    #except OverFlowError:
    #    exp_data = float('inf')
    score = (1 / (1+exp_data))
    return score


geomean_success_list = []
success_ratio = [[] for i in range(len(success_list))]
result_ratio = [0 for i in range(len(result_list))]
rt_list = [[] for i in range(len(result_list))]
for i in range(test):
    ratio = sum(success_list[i]) / num_workload[i]
    success_ratio[i].append(ratio)
    for r in range(len(result_list[i])):
        rt_score = calc_rt_score(target_list[i][r], result_list[i][r])
        result_ratio[i] += rt_score
        rt_list[i].append(rt_score)
    result_ratio[i] /= num_workload[i]

print(success_ratio)
print("real time score: ",result_ratio)
#product = [drop_list[i] * result_ratio[i] for i in range(len(result_ratio))]
#print(product)
workload_dict = [[0] * len(app_name)]
type_dict_list = []
result_dict_list = []
model_qoe_score = [[] for i in range(len(name_list))]
model_rt_score = [[] for i in range(len(name_list))]
for i in range(len(name_list)):
    #print(workload_type_list[i], rt_list[i])
    type_dict = {}
    result_dict = {}
    for key, value in zip(workload_type_list[i], rt_list[i]):
        if key in type_dict:
            type_dict[key] += 1
            result_dict[key] += value
        else:
            type_dict[key] = 1
            result_dict[key] = value
    for key in type_dict:
        ratio = result_dict[key] / type_dict[key]
        model_rt_score[i].append(ratio)
        #print(key, drop_list[i][key])
        model_qoe_score[i].append((type_dict[key] / (drop_list[i][key] + type_dict[key])) * (1-DROP))
    type_dict_list.append(type_dict)
    result_dict_list.append(result_dict)

print("model rt score:", model_rt_score)
print("model qoe score: ",model_qoe_score)
model_overall_score = [[a*b for a,b in zip(model_rt_score[i], model_qoe_score[i])] for i in range(len(model_rt_score))]
print("model overall score: ", model_overall_score)
average_rt = [sum(m) / len(m) for m in model_rt_score]
average_qoe = [sum(m) / len(m) for m in model_qoe_score]
average_overall = [sum(m) / len(m) for m in model_overall_score]
print("average rt score:", average_rt)
print("average qoe score: ",average_qoe)
print("average overall score: ",average_overall)

total_list = [[average_rt[i], average_qoe[i], average_overall[i]] for i in range(len(average_rt))]


SMALL_SIZE = 16
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

metric_name = ['Real-time(RT)\nScore', 'QoE\nScore', 'Overall\nScore']
c = ['#Bda493', '#9a7c66', '#a4d1e1', '#0e465b']

x = np.arange(3)

fig= plt.figure(figsize=(7, 4))
ax = fig.add_subplot(111)
axis_adjust = 0.8 / len(total_list)
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
x_adjust = axis_adjust * (int(len(total_list)/2)-0.5)
for i in range(len(total_list)):
    plt.bar(x-x_adjust+axis_adjust*i, total_list[i], width=axis_adjust, color = c[i], label=test_name[i])
#plt.xlabel('(a) Crossbar', fontsize=BIGGER_SIZE+1, family='Times New Roman')
xticks = np.asarray([i for i in range(len(metric_name))])
plt.xticks(xticks, metric_name, fontsize=MEDIUM_SIZE-1)
plt.yticks(np.arange(0, 1.05, step=0.5), np.arange(0, 1.05, step=0.5).round(decimals=1), fontsize=MEDIUM_SIZE)
plt.ylabel('Score', fontsize=MEDIUM_SIZE, fontweight='bold')
#ax.tick_params(axis='y', which='both', labelsize=MEDIUM_SIZE)

plt.legend(fontsize=12, ncols=4, loc='upper center', bbox_to_anchor=(0.5, 1.1))
plt.tight_layout()
#plt.show()
plt.savefig(soc_type[hw]+'_xr.png')


