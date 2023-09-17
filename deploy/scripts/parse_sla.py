import sys
import re
import csv
import numpy as np
from matplotlib import pyplot as plt

#END_POINT = 170
QOS = 3

wset = int(sys.argv[2])
hw = int(sys.argv[3])

soc_type = ['crossbar', 'noc']

#BASE_END_POINT = 120
#END_POINT = 120
BASE_END_POINT = 170
END_POINT = 170


num_workload = [[] for i in range(QOS)]
num_core = 10
app_name = ['Alex', 'Res', 'YOLO', 'Goo', 'KWS', 'Res18', 'Squeeze', 'YOLO-Lite', 'BERT-base', 'BERT-small']
total_workload_type = list(range(len(app_name)))
#total_priority_type = list(range(12))
target_scale = 1.0

name_list = []
workload_type_list = [[] for i in range(QOS)]
priority_list = [[] for i in range(QOS)]
target_list = [[] for i in range(QOS)]
result_list = [[] for i in range(QOS)]
success_list = [[] for i in range(QOS)]

acquire_list = [[] for i in range(QOS)]
release_list = [[] for i in range(QOS)]

sp_cycles = [0 for i in range(len(app_name))]
#success = []
#ToDo: mp/sp
def g_mean(x):
    a = np.log(x)
    return np.exp(a.mean())
#thread_cycle_storage = [[0 for i in range(len(core_count))] for j in range(num_app)]
#print(num_app, thread_cycle_storage)
#total_cycle_storage = [[0 for i in range(len(core_count))] for j in range(num_app)]

#stat, dynam
name_moca_regex = r"test_moca_set(\d)\b"
name_all_regex = r"test_all_set(\d)\b"
name_comp_regex = r"test_comp_set(\d)\b"
name_vel_regex = r"test_vel_set(\d)\b"
name_noc_regex = r"test_noc_set(\d)\b"
name_nocall_regex = r"test_nocall_set(\d)\b"
start_qos_regex = r"start target scale round (\d)\b"
end_qos_regex = r"end of target (\d) test\b"
end_regex = r"end of test\b"
#profile_regex = r"mode 1 sp runtime profile for workload (\d) number of array 4"
profile_cycle_regex = r"workload (\d) total runtime: (\d*)\b"

type_regex = r"queue id (\d*) workload type: (\d*)\b"
#priority_regex = r"queue id (\d*) priority: (\d*)\b"
#all_regex = r"queue id (\d*) qos: (\d*)\b"
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
#workload_set = str(sys.argv[2])
#for debugging
total_releases = 0
total_acquires = 0

workload_set = ['mixed', 'light', 'heavy'] # 0, 1, 2
workload_set_show = ['C', 'A', 'B']
set_name = None
set_name_show = None
enable = False
set_name = workload_set[wset]
set_name_show = workload_set_show[wset]
print("parsing workload set ", set_name, set_name_show)
end_point = END_POINT
enable = False

with open(sys.argv[1], "r") as f:
    for line in f.readlines():
        name_nocall_search = re.search(name_nocall_regex, line)
        name_noc_search = re.search(name_noc_regex, line)
        name_vel_search = re.search(name_vel_regex, line)
        name_all_search = re.search(name_all_regex, line)
        name_comp_search = re.search(name_comp_regex, line)
        name_moca_search = re.search(name_moca_regex, line)
        start_qos_search = re.search(start_qos_regex, line)
        end_qos_search = re.search(end_qos_regex, line)
        end_search = re.search(end_regex, line)
        type_search = re.search(type_regex, line)
        #priority_search = re.search(priority_regex, line)
        #all_search = re.search(all_regex, line)
        target_search = re.search(target_regex, line)
        result_search = re.search(result_regex, line)
        #overhead_search = re.search(overhead_regex, line)
        release_search = re.search(release_regex, line)
        acquire_search = re.search(acquire_regex, line)
        profile_cycle_search = re.search(profile_cycle_regex, line)
        #profile_search = re.search(profile_regex, line)

        # for profile
        w_prof_type = None
        if profile_cycle_search:
            w_prof_type = int(profile_cycle_search.group(1))
            w_cycle = int(profile_cycle_search.group(2))
            sp_cycles[w_prof_type] = w_cycle
            #print(w_prof_type, w_cycle)
            w_prof_type = None

        if name_comp_search or name_all_search or name_vel_search or name_moca_search or name_noc_search or name_nocall_search:
            name_num = None
            name = None
            if name_all_search:
                name_num = int(name_all_search.group(1))
                #set_name = workload_set[name]
                name = test_name[3]+'_'+workload_set[name_num]
                #name_print = name_search.group(1)+'_workload'+name_search.group(2)+'_qos'+name_search.group(3)
                print(name)
                end_point = END_POINT
            if name_comp_search:
                name_num = int(name_comp_search.group(1))
                #set_name = workload_set[name_num]
                name = test_name[2]+'_'+workload_set[name_num]
                #name_print = name_search.group(1)+'_workload'+name_search.group(2)+'_qos'+name_search.group(3)
                print(name)
                end_point = END_POINT
            if name_moca_search:
                name_num = int(name_moca_search.group(1))
                #set_name = workload_set[name_num]
                name = test_name[1]+'_'+workload_set[name_num]
                #name_print = name_search.group(1)+'_workload'+name_search.group(2)+'_qos'+name_search.group(3)
                print(name)
                end_point = BASE_END_POINT
            if name_vel_search:
                name_num = int(name_vel_search.group(1))
                name = test_name[0]+'_'+workload_set[name_num]
                #name_print = name_search.group(1)+'_workload'+name_search.group(2)+'_qos'+name_search.group(3)
                print(name)
                end_point = BASE_END_POINT
            if name_noc_search:
                name_num = int(name_noc_search.group(1))
                name = test_name[4]+'_'+workload_set[name_num]
                #name_print = name_search.group(1)+'_workload'+name_search.group(2)+'_qos'+name_search.group(3)
                print(name)
                end_point = END_POINT
            if name_nocall_search:
                name_num = int(name_nocall_search.group(1))
                name = test_name[5]+'_'+workload_set[name_num]
                #name_print = name_search.group(1)+'_workload'+name_search.group(2)+'_qos'+name_search.group(3)
                print(name)
                end_point = END_POINT

            if name_num == wset:
                test += 1
                enable = True
                name_list.append(name)
            else:
                enable = False
        elif start_qos_search:
            # initialize
            workload_type = []
            #priority = []
            target = []
            result = []
            num_acquire = 0
            num_release = 0
        elif end_qos_search:
            if enable == True:
                qos = int(end_qos_search.group(1))
                num_workload[qos].append(len(workload_type))
                workload_type_list[qos].append(workload_type)
                #priority_list[qos].append(priority)
                target_list[qos].append(target)
                result_list[qos].append(result)
                #success = [1 if (y < 0) else 0 for x, y in zip(target, result)]
                success = [1 if (x*target_scale > y and y > 0) else 0 for x, y in zip(target, result)]
                success_list[qos].append(success)
                '''
                acquire_list[qos].append(num_acquire)
                release_list[qos].append(num_release)
                print("number of total acquire: ", num_acquire)
                print("number of total release: ", num_release)
                '''

        elif end_search:
            enable = False
        elif type_search and enable:
            if int(type_search.group(1)) < end_point:
                workload_type.append(app_name[int(type_search.group(2))])

        elif target_search and enable:
            if int(target_search.group(1)) < end_point:
                this_target = int(target_search.group(2))
                target.append(this_target)

        elif result_search and enable:
            if int(result_search.group(1)) < end_point:
                cycle = int(result_search.group(2))
                if cycle == 1:
                    cycle = -1
                result.append(cycle)
        '''
        elif release_search:
            num_release += int(release_search.group(2))
        elif acquire_search:
            num_acquire += int(acquire_search.group(2))
        '''
print(name_list)
# test order: QoS-M, QoS-H, QoS-L
total_pre_sla = []
for q in range(QOS):
    geomean_success_list = []
    success_ratio = [[] for i in range(len(success_list[0]))]
    for i in range(test):
        ratio = sum(success_list[q][i]) / num_workload[q][i]
        success_ratio[i].append(ratio)

    print(success_ratio)
    workload_dict = [[0] * len(app_name)]
    type_dict_list = []
    success_dict_list = []
    for i in range(len(name_list)):
        type_dict = {}
        success_dict = {}
        for key, value in zip(workload_type_list[q][i], success_list[q][i]):
            if key in type_dict:
                type_dict[key] += 1
                success_dict[key] += value
            else:
                type_dict[key] = 1
                success_dict[key] = value
        for key in type_dict:
            ratio = success_dict[key] / type_dict[key]
            success_ratio[i].append(ratio)
        type_dict_list.append(type_dict)
        geomean_success_list.append(g_mean(success_ratio[i][1:]))
        success_dict_list.append(success_dict)
    print(geomean_success_list)
    if q != QOS - 1:
        total_pre_sla.append(geomean_success_list)
    else:
        total_pre_sla.insert(0, geomean_success_list)
print(total_pre_sla)

total_sla = [[total_pre_sla[0][i], total_pre_sla[1][i], total_pre_sla[2][i]] for i in range(len(total_pre_sla[0]))]
print(total_sla)

xlabels = ['Veltair', 'MoCA', 'AuRORA-Compute', 'AuRORA-All']
c = ['#Bda493', '#9a7c66', '#a4d1e1', '#0e465b']

SMALL_SIZE = 13
MEDIUM_SIZE = 16
BIGGER_SIZE = 17


fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111)
xticks = ["QoS-L", "QoS-M", "QoS-H"]
x = np.arange(len(xticks))
axis_adjust = 0.75/len(xlabels)#0.3
#plt.bar(x, success_ratio[1], width=axis_adjust, label=xlabels[1], color=c[1])
for i in range(len(total_sla)):
    plt.bar(x+axis_adjust*i, total_sla[i], width=axis_adjust, label=xlabels[i], color=c[i])
#plt.bar(x-axis_adjust, success_ratio[0], width=axis_adjust, label=xlabels[0], color=c[0])
plt.xlabel('QoS', fontsize=MEDIUM_SIZE)
plt.xticks(x+axis_adjust*1.5, xticks)
plt.ylabel('SLA Satisfaction Rate', fontsize=MEDIUM_SIZE)

ax.tick_params(axis='y', which='both', labelsize=MEDIUM_SIZE)
ax.tick_params(axis='x', which='both', labelsize=MEDIUM_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
handles, labels = plt.gca().get_legend_handles_labels()
#add legend to plot
#plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper right', ncol = len(order))

order = [0, 1, 2, 3] #[3, 2, 1, 0]
handles, labels = plt.gca().get_legend_handles_labels()
#add legend to plot
legend_properties = {}#{'weight':'bold'}
leg=plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper center', ncol = 4, prop=legend_properties, frameon=False)
# Get the bounding box of the original legend
bb = leg.get_bbox_to_anchor().transformed(ax.transAxes.inverted())

# Change to location of the legend.
xOffset = 0 #-0.25
yoffset = 0.16#0.25#0.12
#bb.x0 += xOffset
bb.y1 += yoffset
bb.x1 += xOffset
leg.set_bbox_to_anchor(bb, transform = ax.transAxes)

#plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig(soc_type[hw] + '_sla_'+set_name_show+'.png')

