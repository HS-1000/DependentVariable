from collections import defaultdict, deque
import inspect

class DependentStates:
	_class_keywords = [
		"_attrs", "dependencies", "unex_dependencies", 
		"dependency_testing", "validation_node", "updated"
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

	def __getattr__(self, name):
		if name in self._attrs:
			# self.dependency_testing 이 켜져있다면 접근되는 name기록
			if self.dependency_testing:
				self.dependencies[self.dependency_testing].add(name)
			elif self.validation_node:
				if name not in self.dependencies[self.validation_node]:
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
			if name in self._attrs:
				if self._attrs[name].is_origin:
					self._attrs[name].value = value
					self.graph_update()
				else:
					print("This value can only be set via 'update'")
					return False
			else:
				self._attrs[name] = DependentVariable(None, is_origin=True)
				self.dependencies[name] = set()
				is_func = inspect.isfunction(value)
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
		# topological sort
		graph = defaultdict(list)
		indegree = defaultdict(int)
		for node, deps in self.dependencies.items():
			for dep in deps:
				graph[dep].append(node)
				indegree[node] += 1
			if node in indegree:
				indegree[node] = 0
		indegree_0 = deque([node for node in indegree if indegree[node] == 0])
		order = []
		while indegree_0:
			current = indegree_0.popleft()
			order.append(current)
			for child in graph[current]:
				indegree[child] -= 1
				if indegree[child] == 0:
					indegree_0.append(child)
		# Circular dependency checking
		if len(order) == len(indegree):
			return order
		else:
			print("A circular dependency exists")
			return False

	def set_dependencies(self, name, update_func, inplace=True):
		self.dependency_testing = name
		r = update_func(self)
		self.dependency_testing = False
		if inplace:
			self._attrs[name].value = r
		if len(self.dependencies[name]):
			self._attrs[name].is_origin = False
		# 여기서 업데이트 실행하면서 __getattr__에서 이 업데이트 함수가
		# 의존하는 다른 인스턴스들에 이 객체 링크

	def need_update(self, name):
		for d in self.dependencies[name]:
			d = self._attrs[d]
			if d.before == d.value:
				continue
			else:
				return True	
		return False

	def graph_update(self):
		order = self.update_order()
		self.updated = []
		self.reset_unex_dependencies()
		for o in order:
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
		self.validation_node = False

	def reset_unex_dependencies(self):
		self.unex_dependencies = {
			"name" : None,
			"dependencies" : set()
		}

class DependentVariable:
	def __init__(self, value, is_origin=False):
		self.before = []
		self.is_origin = is_origin
		self._value = value

	def update(self, states):
		pass

	def __setattr__(self, name, value):
		if name == "value":
			self.before.append(self._value)
			self._value = value
			if len(self.before) > 10:
				self.before = self.before[1:]
		else:
			super().__setattr__(name, value)

	def __getattr__(self, name):
		if name == "value":
			return self._value
		else:
			return super().__getattr__(name)

	def undo(self):
		# tmp_before = self.before[:-1]
		# self._value = self.before[-1]
		# self.before = tmp_before
		self.before, self._value = self.before[:-1], self.before[-1]
		# del tmp_before
