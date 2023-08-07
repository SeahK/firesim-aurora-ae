import os
import sys
import re
import csv
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
import statistics

def Average(lst):
    return sum(lst) / len(lst)

wset = int(sys.argv[2])
hw = int(sys.argv[3])

soc_type = ['crossbar', 'noc']

#BASE_END_POINT = 120
#END_POINT = 180

BASE_END_POINT = 140
END_POINT = 140

QOS = 3
num_workload = [[] for i in range(QOS)]
num_core = 10
app_name = ['Alex', 'Res', 'YOLO', 'Goo', 'KWS', 'Res18', 'Squeeze', 'YOLO-Lite', 'BERT-base', 'BERT-small']

total_workload_type = list(range(len(app_name)))
total_priority_type = list(range(12))
target_scale = 1.0

test_name = ["veltair", "MoCA", "Comp", "All"]

name_list = []
workload_type_list = [[] for i in range(QOS)]
target_list = [[] for i in range(QOS)]
result_list = [[] for i in range(QOS)]

acquire_list = [[] for i in range(QOS)]
release_list = [[] for i in range(QOS)]


# single program time
sp_cycles = [0] * len(app_name)
#success = []
#ToDo: mp/sp

name_moca_regex = r"test_moca_set(\d)\b"
name_all_regex = r"test_all_set(\d)\b"
name_comp_regex = r"test_comp_set(\d)\b"
name_vel_regex = r"test_vel_set(\d)\b"
name_noc_regex = r"test_noc_set(\d)\b"
name_nocqos_regex = r"test_nocqos_set(\d)\b"
start_all_regex = r"start target scale round (\d)\b"
end_all_regex = r"end of target (\d) test\b"
end_regex = r"end of test\b"
#profile_regex = r"mode 1 sp runtime profile for workload (\d) number of array 4"
profile_cycle_regex = r"workload (\d) total runtime: (\d*)\b"

type_regex = r"queue id (\d*) workload type: (\d*)\b"
#priority_regex = r"queue id (\d*) priority: (\d*)\b"
#qos_regex = r"queue id (\d*) qos: (\d*)\b"
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
#qos = []
#priority = []
target = []
result = []
#workload_set = str(sys.argv[2])
workload_set = ['mixed', 'light', 'heavy'] # 0, 1, 2
workload_set_show = ['C', 'A', 'B']
set_name = None
set_name_show = None
enable = False
set_name = workload_set[wset]
set_name_show = workload_set_show[wset]
print("parsing workload set ", set_name, set_name_show)

#qos_type = ['l', 'm', 'h']
with open(sys.argv[1], "r") as f:
    for line in f.readlines():
        name_nocqos_search = re.search(name_nocqos_regex, line)
        name_noc_search = re.search(name_noc_regex, line)
        name_vel_search = re.search(name_vel_regex, line)
        name_all_search = re.search(name_all_regex, line)
        name_comp_search = re.search(name_comp_regex, line)
        name_moca_search = re.search(name_moca_regex, line)
        start_all_search = re.search(start_all_regex, line)
        end_all_search = re.search(end_all_regex, line)
        end_search = re.search(end_regex, line)
        type_search = re.search(type_regex, line)
        #priority_search = re.search(priority_regex, line)
        #qos_search = re.search(qos_regex, line)
        target_search = re.search(target_regex, line)
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
            #print(w_prof_type, w_cycle)
            sp_cycles[w_prof_type] = w_cycle
            w_prof_type = None

        if name_comp_search or name_all_search or name_vel_search or name_moca_search or name_noc_search or name_nocqos_search:
            name = None
            name_num = None
            if name_all_search:
                name_num = int(name_all_search.group(1))
                name = test_name[3]+'_'+workload_set[name_num]
                #name_print = name_search.group(1)+'_workload'+name_search.group(2)+'_qos'+name_search.group(3)
                print(name)
                end_point = END_POINT
            if name_comp_search:
                name_num = int(name_comp_search.group(1))
                name = test_name[2]+'_'+workload_set[name_num]
                #name_print = name_search.group(1)+'_workload'+name_search.group(2)+'_qos'+name_search.group(3)
                print(name)
                end_point = END_POINT
            if name_moca_search:
                name_num = int(name_moca_search.group(1))
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
            if name_nocqos_search:
                name_num = int(name_nocqos_search.group(1))
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

        elif start_all_search:
            # initialize
            workload_type = []
            #priority = []
            target = []
            result = []

        elif end_all_search:
            if enable == True:
                qos = int(end_all_search.group(1))
                num_workload[qos].append(len(workload_type))
                workload_type_list[qos].append(workload_type)
                target_list[qos].append(target)
                result_list[qos].append(result)

        elif end_search:
            enable = False

        elif type_search and enable:
            if int(type_search.group(1)) < end_point:
                w_type = int(type_search.group(2))
                workload_type.append(w_type)
                #workload_type.append(app_name[int(type_search.group(2))])

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

print(name_list)

# ToDo: have to sort baseline names on final plot
#result_list = [x for _,x in sorted(zip(name_list, result_list))]
#workload_type_list = [x for _,x in sorted(zip(name_list, workload_type_list))]
#priority_list = [x for _,x in sorted(zip(name_list, priority_list))]
#name_list.sort()

all_list = [[[[] for i in range(len(app_name))] for j in range(len(name_list))] for q in range(QOS)]
#priority_all_list = [[[] for i in range(len(app_name))] for j in range(len(name_list))]
for q in range(QOS):
    for i in range(len(name_list)):
        for j in range(len(workload_type_list[q][i])):
            workload_type = workload_type_list[q][i][j]
            all_list[q][i][workload_type].append(result_list[q][i][j])

#print(all_list)
#print(priority_all_list)
average_all = [[[] for i in range(len(name_list))] for q in range(QOS)]
median_all = [[[] for i in range(len(name_list))] for q in range(QOS)]

#priority_average_all = [[] for i in range(len(name_list))]

for q in range(QOS):
    for i in range(len(all_list[q])):
        #average_all[q][i].append(0)
        #median_all[q][i].append(0)
        #priority_average_all[i].append(0)
        for j in range(len(all_list[q][i])):
            if len(all_list[q][i][j]) != 0:
                average_all[q][i].append(Average(all_list[q][i][j]))
                median_all[q][i].append(statistics.median(all_list[q][i][j]))
                #priority_average_all[i].append(Average(priority_all_list[i][j]))
            else:
                average_all[q][i].append(0)
                median_all[q][i].append(0)
                #priority_average_all[i].append(0)

#print(priority_average_all)
#fairness: min((Csp/Cmp) / 1) -> (same priority)
#STP: sum(Csp/Cmp)

# test order: QoS-M, QoS-H, QoS-L
total_pre_fairness = []
total_pre_stp = []
for q in range(QOS):
    raw_fair_list = []
    raw_stp_list = []

    pp = []
    for i in range(len(name_list)):
        pp = []
        stp = 0
        for j in range(len(app_name)):
            #print(average_all[q][i])
            if average_all[q][i][j] != 0:
                ppi = sp_cycles[j] / average_all[q][i][j]
                #print(i, j, sp_cycles[j], average_all[q][i][j])
                stp += ppi
                p_factor = 1 # all same priority
                #p_factor = priority_average_all[i][j] / sum(priority_average_all[i])
                pp.append(ppi/p_factor)
        min_fair = 1
        pp_list = []
        #print("pp for qos ", q, pp)
        for j in range(len(pp)):
            for k in range(len(pp)):
                if pp[j] != 0 and pp[k] != 0:
                    min_fair = min(min_fair, pp[j] / pp[k])
                    pp_list.append(pp[j]/pp[k])

        #print(pp_list)
        #min_fair = sum(pp_list)/len(pp_list)
        raw_fair_list.append(min_fair)
        raw_stp_list.append(stp)
        raw_fair_list.sort()
        raw_stp_list.sort()

    fair_list = raw_fair_list
    stp_list = raw_stp_list
    #print("for qos ", q)
    #print(fair_list)
    #print(stp_list)
    #print(name_list)
#name_list = [x for _,x in sorted(zip(qos_list, name_list))]
#fair_list = [x for _,x in sorted(zip(qos_list, fair_list))]
#stp_list = [x for _,x in sorted(zip(qos_list, stp_list))]

    fair_norm_list = [i / fair_list[0] for i in fair_list]
    stp_norm_list = [i / stp_list[0] for i in stp_list]


    #print(fair_norm_list)
    #print(stp_norm_list)
    if q != QOS - 1:
        total_pre_fairness.append(fair_norm_list)
        total_pre_stp.append(stp_norm_list)
    else:
        total_pre_stp.insert(0, stp_norm_list)
        total_pre_fairness.insert(0, fair_norm_list)

print(total_pre_fairness)
print(total_pre_stp)
total_fairness = [[total_pre_fairness[0][i], total_pre_fairness[1][i], total_pre_fairness[2][i]] for i in range(len(total_pre_fairness[0]))]
print(total_fairness)
total_stp = [[total_pre_stp[0][i], total_pre_stp[1][i], total_pre_stp[2][i]] for i in range(len(total_pre_stp[0]))]
print(total_stp)


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
for i in range(len(total_fairness)):
    plt.bar(x+axis_adjust*i, total_fairness[i], width=axis_adjust, label=xlabels[i], color=c[i])
#plt.bar(x-axis_adjust, success_ratio[0], width=axis_adjust, label=xlabels[0], color=c[0])
plt.xlabel('QoS', fontsize=MEDIUM_SIZE)
plt.xticks(x+axis_adjust*1.5, xticks)
plt.ylabel('Normalized Fairness', fontsize=MEDIUM_SIZE)

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
plt.savefig(soc_type[hw]+'_fairness_'+set_name_show+'.png')


fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111)
xticks = ["QoS-L", "QoS-M", "QoS-H"]
x = np.arange(len(xticks))
axis_adjust = 0.75/len(xlabels)#0.3
#plt.bar(x, success_ratio[1], width=axis_adjust, label=xlabels[1], color=c[1])
for i in range(len(total_stp)):
    plt.bar(x+axis_adjust*i, total_stp[i], width=axis_adjust, label=xlabels[i], color=c[i])
#plt.bar(x-axis_adjust, success_ratio[0], width=axis_adjust, label=xlabels[0], color=c[0])
plt.xlabel('QoS', fontsize=MEDIUM_SIZE)
plt.xticks(x+axis_adjust*1.5, xticks)
plt.ylabel('Normalized STP', fontsize=MEDIUM_SIZE)

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
plt.savefig(soc_type[hw]+'_stp_'+set_name_show+'.png')


