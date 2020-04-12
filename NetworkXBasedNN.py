import networkx as nx
import numpy as np 
import matplotlib.pyplot as plt
import datetime

#本案例实现了最简单的深层前馈神经网络的前向传播过程，其中数据可以自定义，传播过程和异步可以自定义，无需同步

class LIF:
	'''
	params
	gE：总兴奋电导
	VE：反电势
	VL：漏电势
	TaoE​：兴奋神经元电导延迟
	dt：时间步长
	V：膜电位，初始为-40（reset）
	Vrest：静息电位，为-40
	TaoM：时间常数，大约为10ms
	gL：漏电电导
	Threshold：阈值，大于阈值才激活，否则无响应
	'''
	def __init__(self):
		self.gE = 15.4
		self.gL = 0.5
		self.VE = -23.045
		self.VL = 1.036
		self.TaoE = 0.01
		self.lasttime = datetime.datetime.now()
		self.Vrest = -40
		self.V = self.Vrest
		self.TaoM = 0.01
		self.Threshold = -20

	def update(self,inputs,outputs):
		'''
		inputs表示上一神经元的输入数组
		outputs输出神经元构成的列表
		'''
		try:
			if inputs.any():
				self.weight = np.random.randn((len(inputs)))
				self.inputs = np.dot(inputs,self.weight)
				self.dt = (datetime.datetime.now()-self.lasttime).total_seconds()
				self.lasttime = datetime.datetime.now()
				self.gE = self.gE - self.gE*self.dt/self.TaoE + self.inputs
				self.V = self.V - self.dt/self.TaoM*(self.V-self.VL+self.gE/self.gL*(self.V-self.VE))
				self.spiking = self.V > self.Threshold
				if self.spiking:
					self.weight_out = np.random.randn((len(outputs)))
					return self.V*self.weight_out
				else:
					self.V = self.Vrest
		except:
			return None


class Test:
	def __init__(self,name):
		self.name = name
		self.data = np.random.randn()

	def receive(self,G):
		if list(G.predecessors(self.name)):
			#采用加权的方式来直接相加
			for last_nu in list(G.predecessors(self.name)):
				self.data += G.edges[last_nu,self.name]['weight']*G.nodes[last_nu]["object"].data
				self.send(G)
		else:
			pass

	def send(self,G):
		if list(G.successors(self.name)):
			#通知下游神经元进行信号接收
			for next_nu in list(G.successors(self.name)):
				G.nodes[next_nu]["object"].receive(G)
		else:
			pass

def Linear(G,num_last,num_this):
	list_name = []
	try:
		last_layer = eval(list(G.nodes)[-1][list(G.nodes)[-1].find("["):list(G.nodes)[-1].find("]")+1])[0]
		#创建顶点
		G.add_nodes_from(["Layer"+str([last_layer+1])+str(i) for i in range(num_this)])
		#实例化对象
		for point in range(num_this):
			G.nodes["Layer"+str([last_layer+1])+str(point)]["object"] = Test("Layer"+str([last_layer+1])+str(point))
		#添加权重边（随机初始化权重）全连接层
		for index1 in range(num_last):
			for index2 in range(num_this):
				G.add_weighted_edges_from([("Layer"+str([last_layer])+str(index1),("Layer"+str([last_layer+1])+str(index2)),np.random.randn())])
		return G
	except:
		#是首层：先创建第一层
		G.add_nodes_from(["Layer[0]"+str(i) for i in range(num_last)])
		for point in range(num_last):
			G.nodes["Layer"+str([0])+str(point)]["object"] = Test("Layer"+str([0])+str(point))

		#再创建第二层
		G.add_nodes_from(["Layer[1]"+str(i) for i in range(num_this)])
		for point in range(num_this):
			G.nodes["Layer"+str([1])+str(point)]["object"] = Test("Layer"+str([1])+str(point))

		#再创建权重边
		last_layer = 0
		for index1 in range(num_last):
			for index2 in range(num_this):
				G.add_weighted_edges_from([("Layer"+str([last_layer])+str(index1),("Layer"+str([last_layer+1])+str(index2)),np.random.randn())])
		return G

#定义网络结构，Linear的方式和nn是一样的，区别在于没有sigmoid（SNN不需要sigmoid）
G = nx.DiGraph()
G = Linear(G,4,6)
G = Linear(G,6,5)
G = Linear(G,5,2)
G = Linear(G,2,1)

'''
#网络可视化
plt.plot()
nx.draw(G, with_labels=True, font_weight='bold')
plt.show()'''

#前馈运算
for i in range(4): #第一层全部的神经元前馈运算，当然也可以指定某几个进行前馈运算
	G.nodes["Layer[0]"+str(i)]["object"].send(G)

print(G.nodes[list(G.nodes)[-1]]["object"].data)