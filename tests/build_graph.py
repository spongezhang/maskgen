import json
import networkx as nx
import argparse
import sys
from networkx.readwrite import json_graph
import random

#load selection plugin, operations and final operations
with open('../tests/top_operation.json') as data_file:    
    top_operation = json.load(data_file)['operation']
with open('../tests/operation_pool.json') as data_file:
    json_data = json.load(data_file)
    operation = json_data['operation']
    json_data.pop('operation', None)
with open('../tests/final_operation.json') as data_file:    
    final_operation = json.load(data_file)['operation']

#number of base+donor, 1 mean 1 base, 2 means 1 base + 1 donor 
top_number = random.randint(3, 3)
#list of top nodes, including select image and convert to png
top_list = []
#top edge list
top_edge_list = []
#node id number
cur_id = 0
for i in range(top_number):
    cur_list = []
    cur_edge_list = []

    #i==0 base branch, otherwise donor branch
    if i == 0:
        cur_node = top_operation[0].copy()
    else:
        cur_node = top_operation[1].copy()
    cur_node[u'id'] = str(cur_id).decode('utf-8')
    cur_id = cur_id+1
    cur_list.append(cur_node)
    
    #toPNG plugin 
    cur_node = top_operation[2].copy()
    cur_node[u'id'] = str(cur_id).decode('utf-8')
    cur_list.append(cur_node)
    cur_edge_list.append((cur_id-1,cur_id))
    cur_id = cur_id+1
    
    # mask selection plugin
    if i>0:
        cur_node = top_operation[3].copy()
        cur_node[u'id'] = str(cur_id).decode('utf-8')
        cur_list.append(cur_node)
        cur_edge_list.append((cur_id-1,cur_id))
        cur_id = cur_id+1
        
    top_list.append(cur_list)
    top_edge_list.append(cur_edge_list)

print(top_edge_list)

#maximum length of the base branch
total_length = random.randint(len(top_list)-1+1,len(top_list)-1+3)

#decide splice location in the base branch
splice_paste_list = []
for i in range(len(top_list)-1):
    splice_point = random.randint(1,total_length-1)
    while splice_point in splice_paste_list:
        splice_point = random.randint(1,total_length-1)
    splice_paste_list.append(splice_point)

#print(splice_paste_list)

#refence node for local operation
source_ref_id = 1
target_ref_id = 1

#total list for nodes and edges
node_list = []
edges_list = []

#index of the top branch
top_ref = 0
#build base branch
node_list = node_list + top_list[0]
edges_list.append({u'source':0,u'target':1})
top_ref = top_ref + 1
pre_id = 1

#build the graph
for i in range(total_length):
    #find splice position
    if i in splice_paste_list:
        source_ref_id = int(edges_list[-1]['target'])
        targe_ref_id = cur_id

        #insert donor branch
        node_list = node_list + top_list[top_ref]
        for j in top_edge_list[top_ref]:
            edges_list.append({u'source':int(j[0]),u'target':int(j[1])})
        end_id = top_edge_list[top_ref][-1][1]
        splice_paste_node = operation[0].copy()
        splice_paste_node['id'] = str(cur_id).decode('utf-8')
        node_list.append(splice_paste_node)
        #base link
        edges_list.append({u'source':int(pre_id),u'target':int(cur_id)})
        #donor link
        edges_list.append({u'source':int(end_id),u'target':int(cur_id)})
        top_ref = top_ref+1

    #other operations
    else:
        pick_operation_idx = random.randint(1,len(operation)-1)
        #can't use local operation, no valid mask
        if source_ref_id == target_ref_id:
            while 'Local' in operation[pick_operation_idx]['plugin']:
                pick_operation_idx = random.randint(1,len(operation)-1)
        #Deal with local operation
        if 'Local' in operation[pick_operation_idx]['plugin']:
            cur_node = operation[pick_operation_idx].copy()
            cur_node[u'id'] = str(cur_id).decode('utf-8')

            #set mask source and target 
            cur_node[u'arguments'][u'inputmaskname'][u'source'] = str(source_ref_id)
            cur_node[u'arguments'][u'inputmaskname'][u'target'] = str(target_ref_id)
            node_list.append(cur_node)
            edges_list.append({u'source':int(pre_id),u'target':int(cur_id)})
        else:
            cur_node = operation[pick_operation_idx].copy()
            cur_node[u'id'] = str(cur_id).decode('utf-8')
            node_list.append(cur_node)
            edges_list.append({u'source':int(pre_id),u'target':int(cur_id)})
            #image size changes, local filter can't be used. One can add others
            if operation[pick_operation_idx][u'plugin'] == u'Crop':
                cur_ref_id = next_ref_id
    
    pre_id = cur_id
    cur_id = cur_id+1 

#deal with the final operation(CompressAs)
cur_node = final_operation[0]
cur_node[u'id'] = str(cur_id).decode('utf-8')
node_list.append(cur_node)
edges_list.append({u'source':int(cur_id-1),u'target':int(cur_id)})

#sort node based on id
node_list.sort(key = lambda x:int(x[u'id']))
edges_list.sort(key = lambda x:int(x[u'target']))

#get final graph and journal file
json_data[u'nodes'] = node_list
json_data[u'links'] = edges_list

print(node_list)
print(edges_list)

G = json_graph.node_link_graph(json_data, multigraph=False, directed=True)
with open('generated.json', 'w') as fp:
    json.dump(json_data, fp,indent=4, separators=(',', ': '))


print(G.nodes())
print(G.edges())

