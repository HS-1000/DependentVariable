from collections import defaultdict, deque
import inspect
import numpy as np
import pandas as pd
import traceback

class DependentStates:
	_class_keywords = [
		"_attrs", "dependencies", "unex_dependencies", 
		"dependency_testing", "validation_node", "updated",
		"independent"
	]

	def __init__(self):
		self._attrs = {}
		self.dependencies = {}
		self.unex_dependencies = {
			"name" : None,
			"dependencies" : set()
		}
		self.dependency_testing = False
		self.validation_node = False
		self.updated = []
		self.independent = {}

	def __getattr__(self, name):
		if name in self._attrs:
			# self.dependency_testing 이 켜져있다면 접근되는 name기록
			if self.dependency_testing:
				if name != self.dependency_testing:
					self.dependencies[self.dependency_testing].add(name)
			elif self.validation_node:
				if name not in (*self.dependencies[self.validation_node], self.validation_node):
					self.unex_dependencies["name"] = self.validation_node
					self.unex_dependencies["dependencies"].add(name)
			return self._attrs[name].value
		elif name in self.__dict__:
			return self.__dict__[name]
		else:
			raise AttributeError(f"name '{name}' is not defined")
	
	def __setattr__(self, name, value):
		if name in self._class_keywords:
			super().__setattr__(name, value)
		else:
			is_func = inspect.isfunction(value)
			if name in self._attrs:
				if is_func:
					self.dependencies[name] = set()
					self._attrs[name].update = value
					self.set_dependencies(name, value)
					self.graph_update()
				elif self._attrs[name].is_origin:
					self._attrs[name].value = value
					self.graph_update()
				else:
					print("This value can only be set via 'update'")
					return False
			else:
				self._attrs[name] = DependentVariable(None, is_origin=True)
				self.dependencies[name] = set()
				if is_func:
					self._attrs[name].update = value
					self.set_dependencies(name, value)
				else:
					self._attrs[name].value = value

	def __delattr__(self, name):
		for k in self.dependencies.keys():
			if k == name:
				continue
			elif name in self.dependencies[k]:
				print(f"{name} is dependent on another variable")
				return False
		del self.dependencies[name]
		del self._attrs[name]

	def update_order(self):
		# Topological sort를 위한 그래프 및 진입 차수 초기화
		graph = defaultdict(list)
		indegree = defaultdict(int)

		# 의존성과 그래프 구성
		for node, deps in self.dependencies.items():
			for dep in deps:
				graph[dep].append(node)
				indegree[node] += 1  # 노드의 진입 차수 증가
			if node not in indegree:  # 노드가 처음 등장할 때만 0으로 초기화
				indegree[node] = 0

		# 진입 차수가 0인 노드 큐 초기화
		indegree_0 = deque([node for node in indegree if indegree[node] == 0])
		order = []

		# Topological sorting
		while indegree_0:
			current = indegree_0.popleft()
			order.append(current)
			for child in graph[current]:
				indegree[child] -= 1
				if indegree[child] == 0:
					indegree_0.append(child)

		# 순환 의존성 체크
		if len(order) == len(indegree):
			result = []
			for o in order:
				if len(self.dependencies[o]):
					result.append(o)
			return result
		else:
			raise ValueError('A circular dependency exists')

	def set_dependencies(self, name, update_func, inplace=True, ):
		self.dependency_testing = name
		r = update_func(self)
		self.dependency_testing = False
		if inplace:
			self._attrs[name].value = r
		if len(self.dependencies[name]):
			self._attrs[name].is_origin = False

	def need_update(self, name):
		for d in self.dependencies[name]:
			d = self._attrs[d]
			try:
				is_change = not (d.before[-1] == d.value)
			except:
				is_change = True
			if is_change:
				return True
			else:
				continue
		return False

	def graph_update(self):
		order = self.update_order()
		self.updated = []
		self.reset_unex_dependencies()
		for o in order:
			try:
				if self.need_update(o):
					self.validation_node = o
					self._attrs[o].value = self._attrs[o].update(self)
					self.updated.append(o)
					if len(self.unex_dependencies["dependencies"]):
						# undo, reset dependencies
						for u in self.unex_dependencies["dependencies"]:
							self.dependencies[o].add(u)
						for u in self.updated:
							self._attrs[u].undo()
						self.graph_update()
						break
			except Exception as e:
				traceback.print_exc()
				print(e)
				print("\nUpdate error value name:", o)
				exit()
		self.validation_node = False

	def reset_unex_dependencies(self):
		self.unex_dependencies = {
			"name" : None,
			"dependencies" : set()
		}

	def to_dict(self):
		keys_ = self._attrs.keys()
		val_dict = {}
		for k in keys_:
			val_dict[k] = self._attrs[k].value
		return val_dict

class DependentVariable:
	def __init__(self, value, is_origin=False):
		self.before = []
		self.is_origin = is_origin
		self._value = value

	def update(self):
		pass

	def __setattr__(self, name, value):
		if name == "value":
			self.before.append(self.value)
			self._value = value
			if len(self.before) > 10:
				self.before = self.before[1:]
		else:
			super().__setattr__(name, value)

	def __getattr__(self, name):
		if name == "value":
			if isinstance(self._value, list):
				return [*self._value]
			elif isinstance(self._value, dict):
				return {**self._value}
			elif isinstance(self._value, np.ndarray):
				return self._value.copy()
			elif isinstance(self._value, pd.DataFrame):
				return self._value.copy()
			else:
				return self._value
		else:
			return super().__getattr__(name)

	def undo(self):
		self.before, self._value = self.before[:-1], self.before[-1]
