import networkx as nx
import matplotlib.pyplot as plt
import os
import json
import powerlaw
import numpy as np

netname = 'gplus' # twitter or gplus
nodeId = 113597493946570654755
k_sp = 1/1000000000000 # spring layout
k_ns = 0.08 # node size
width = 0.05 # edge width

edge_file_name = netname + str(nodeId)
edge_file_path = "../data/" + netname + "/"+str(nodeId) + ".edges"
output_path = "../output/" + edge_file_name +"/"

if not os.path.exists(output_path):
	os.mkdir(output_path)

# record file size 
fsize = os.path.getsize(edge_file_path)
fsize = "File size: {:.2f} KB".format(fsize/1024)
with open(output_path + edge_file_name + '_info.txt', 'wt') as f:
	print (fsize, file=f)
	print (fsize)

# load the network based on .edges file
g = nx.read_edgelist(edge_file_path, nodetype=int, create_using=nx.DiGraph(),)
# obtain a list of all the nodes
node_list=list(g)

# record original network info
g_info = nx.info(g)
with open(output_path + edge_file_name + '_info.txt', 'at') as f:
	print (g_info, file=f)
	print (g_info)

## draw the original network
sp=nx.spring_layout(g, k= k_sp)
plt.axis('off')
nx.draw_networkx(g,pos=sp,with_labels=False, node_size=[(v+1) * k_ns for v in dict(g.in_degree()).values()], arrows=False, width=width)
plt.savefig(output_path + edge_file_name + '_visial_size_original.svg', format="svg")
# plt.show()
plt.close()

# add the ego node
g.add_node(nodeId)
# add edges for the ego node, it is connected to all other nodes
g.add_edges_from((v,nodeId) for v in node_list)

## update network info
g_info = nx.info(g)
with open(output_path + edge_file_name + '_info.txt', 'at') as f:
	print (g_info, file=f)
	print (g_info)


## draw the ego network
sp=nx.spring_layout(g, k = k_sp)
plt.axis('off')
nx.draw_networkx(g,pos=sp,with_labels=False, node_size=[(v+1) * k_ns for v in dict(g.in_degree()).values()], arrows=False, width=width)
plt.savefig(output_path + edge_file_name + '_visial_size_addEgo.svg', format="svg")
# plt.show()
plt.close()

## compute degree centrality
in_degrees=dict(g.in_degree())
in_degree_values=sorted(set(in_degrees.values()))
histogram=[list(in_degrees.values()).count(i)/float(nx.number_of_nodes(g)) for i in in_degree_values]
indeg_list=in_degrees.values()

## draw the degree
# plt.loglog(in_degree_values, histogram, 'o')
plt.plot(in_degree_values, histogram, 'o')
plt.xlabel("in-degree: k (log-scale)")
plt.ylabel("probability distribution: Pk (log-scale)")
plt.xscale('log')
plt.yscale('log')
plt.title(edge_file_name)
plt.savefig(output_path + edge_file_name + '_indeg.png', format="png")
# plt.show()
plt.close()


# fit the degree 
degree_sequence = sorted([d for d in indeg_list], reverse=True)
fit = powerlaw.Fit(np.array(degree_sequence)+1, discrete=True)
fig1 = fit.plot_pdf(color='b', linewidth=2)
fit.power_law.plot_pdf(color='g', linestyle='--', ax=fig1)
# the alpha here is actually the r in power law 
print('xmin= ', fit.xmin,'alpha= ',fit.power_law.alpha,'  sigma= ',fit.power_law.sigma)
plt.show()

# compare different distributions
R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
print (R, p)
R, p = fit.distribution_compare('power_law', 'stretched_exponential', normalized_ratio=True)
print (R, p)

