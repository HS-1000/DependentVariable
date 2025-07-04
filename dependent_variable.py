from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional, Dict, Set, List
from collections import defaultdict, deque
import inspect
from functools import wraps

@dataclass
class Variable:
	"""노드, 단일 변수와 업데이트 함수"""
	value: Any = None
	compute_func: Optional[Callable] = None

class DependencyGraph:
	"""
	변수들 간의 의존성을 관리하고, 변경된 변수 및 그에 종속된 변수들의
	업데이트 순서를 효율적으로 결정하는 그래프 클래스입니다.
	"""

	def __init__(self):
		# dependencies[A] = {B, C} 의미: 'A'는 'B'와 'C'에 의존합니다.
		self.dependencies: Dict[str, Set[str]] = defaultdict(set)
		
		# dependents[B] = {A, D} 의미: 'A'와 'D'는 'B'에 의존합니다.
		self.dependents: Dict[str, Set[str]] = defaultdict(set)
		
		# is_change[X] = True 의미: 'X' 변수가 변경되었거나, X에 의존하는 변수가 변경되어
		self.is_change: Dict[str, bool] = defaultdict(bool)

	def _mark_as_changed(self, node: str) -> None:
		"""
		주어진 'node'를 '변경됨' True로 표시하고,
		그 노드에 의존하는 모든 노드들까지 True로 표시합니다.
		"""
		if self.is_change[node]: # 이미 변경된 노드
			return

		self.is_change[node] = True
		
		for dependent_node in self.dependents[node]:
			self._mark_as_changed(dependent_node)

	def add_dependency(self, target: str, source: str) -> None:
		"""
		새로운 의존성 관계를 추가합니다: 'target'은 'source'에 의존합니다.
		관계 추가 시, 'target'의 의존성 구조가 변경되었으므로 'target'을 변경 상태로 표시하고 전파합니다.
		"""
		# is_change 딕셔너리에 새로운 노드가 추가될 경우를 대비해 초기화
		# _mark_as_changed가 호출되기 전에 기본값(False)을 설정하여 예상치 못한 동작 방지
		if target not in self.is_change:
			self.is_change[target] = False
		if source not in self.is_change:
			self.is_change[source] = False

		# 실제로 새로운 의존성인 경우에만 추가 및 변경 전파
		if source not in self.dependencies[target]:
			self.dependencies[target].add(source)
			self.dependents[source].add(target) # 역방향 의존성도 함께 기록
			self._mark_as_changed(target) # 'target'의 의존성이 변경되었으므로 'target'을 변경 상태로 표시

	def remove_dependency(self, target: str, source: str) -> None:
		"""
		기존 의존성 관계를 제거합니다: 'target'은 'source'에 더 이상 의존하지 않습니다.
		관계 제거 시, 'target'의 의존성 구조가 변경되었으므로 'target'을 변경 상태로 표시하고 전파합니다.
		"""
		if source in self.dependencies[target]:
			self.dependencies[target].discard(source)
			self.dependents[source].discard(target) # 역방향 의존성도 함께 제거
			self._mark_as_changed(target) # 'target'의 의존성이 변경되었으므로 'target'을 변경 상태로 표시

	def clean_dependency(self, target: str) -> None:
		"""
		특정 변수('target')의 모든 의존성을 제거합니다.
		역방향(dependents['target'])은 유지합니다.
		"""
		target_sources = list(self.dependencies[target])
		for source in target_sources:
			self.remove_dependency(target, source)

	def get_dependency(self, target: str) -> Set[str]:
		"""
		특정 변수('target')가 의존하는 변수들의 집합을 반환합니다.
		defaultdict의 특성상 'target'이 존재하지 않으면 빈 set이 자동으로 생성되어 반환됩니다.
		"""
		return self.dependencies[target]

	def set_changed(self, node: str) -> None:
		"""
		외부에서 특정 변수('node')의 값이 직접 변경되었음을 이 그래프에게 알립니다.
		해당 변수('node')와 그에 직/간접적으로 의존하는 모든 변수들을
		'업데이트 필요' 상태(is_change = True)로 전파합니다.
		"""
		# 노드가 is_change 맵에 존재하는지 확인 (새로운 노드일 수 있음)
		if node not in self.is_change:
			self.is_change[node] = False # 초기값 설정 (False)
		self._mark_as_changed(node) # 변경 상태 전파 시작

	def reset_change_status(self, node: str) -> None:
		"""
		특정 노드('node')의 변경 상태를 '업데이트 완료'(False)로 초기화합니다.
		주로 해당 노드의 업데이트가 성공적으로 완료된 후 호출됩니다.
		"""
		self.is_change[node] = False

	def get_update_order(self) -> List[str]:
		"""
		현재 '업데이트 필요' 상태(is_change = True)인 노드들과
		이들이 의존하는 노드들을 포함하여, 업데이트를 수행해야 하는 순서를
		위상 정렬(Topological Sort)하여 반환합니다.
		변경된 부분만 효율적으로 업데이트하도록 최적화되어 있습니다.
		"""
		# 1. 업데이트가 필요한 모든 '관련된' 노드들을 식별합니다.
		#	이는 is_change가 True인 노드들과, 이들이 업데이트되기 위해 선행되어야 하는 노드들을 포함합니다.
		all_relevant_nodes = set()
		for node, changed in self.is_change.items():
			if changed: 
				all_relevant_nodes.add(node) # 'is_change'가 True인 노드는 무조건 포함
				for source in self.dependencies[node]:
					all_relevant_nodes.add(source)
				# 이 노드('node')가 의존하는 모든 'source' 노드들도 관련 노드로 추가
				# ('source'는 비록 is_change가 False여도 'node' 업데이트의 선행 조건이 될 수 있음)

		# 2. 식별된 '관련된' 노드들만을 포함하는 '부분 그래프'를 구성합니다.
		#	이 부분 그래프 내에서만 위상 정렬을 수행하여 효율성을 높입니다.
		graph_for_topo = defaultdict(list) # 위상 정렬용 그래프: source -> target 관계
		indegree_for_topo = defaultdict(int) # 각 노드의 진입 차수

		for target_node in all_relevant_nodes:
			# 'target_node'가 의존하는 모든 'source_node'들을 찾아 그래프 에지를 구성합니다.
			# (즉, source_node가 먼저 처리되어야 target_node를 처리할 수 있음)
			for source_node in self.dependencies[target_node]:
				if source_node in all_relevant_nodes: # '관련된' 노드들 사이의 의존성만 고려
					graph_for_topo[source_node].append(target_node)
					indegree_for_topo[target_node] += 1
		
		# 3. 진입차수가 0인 노드들(위상 정렬의 시작점)을 큐에 추가합니다.
		#	'all_relevant_nodes'에 속하지만 'indegree_for_topo'에 없는 노드는 진입차수 0으로 간주합니다.
		queue = deque([node for node in all_relevant_nodes if indegree_for_topo[node] == 0])
		
		topo_order_full = [] # 위상 정렬된 모든 '관련된' 노드 (is_change가 False인 노드도 포함될 수 있음)
		num_processed_nodes = 0 # 위상 정렬 처리된 노드의 수
		
		# 4. 큐를 사용하여 위상 정렬을 수행합니다 (Kahn's Algorithm).
		while queue:
			current_node = queue.popleft()
			topo_order_full.append(current_node)
			num_processed_nodes += 1
			
			# 현재 노드가 선행 조건인 모든 후속 노드들의 진입차수를 감소시킵니다.
			for neighbor_node in graph_for_topo[current_node]:
				indegree_for_topo[neighbor_node] -= 1
				# 진입차수가 0이 된 노드는 다음 처리 대상으로 큐에 추가합니다.
				if indegree_for_topo[neighbor_node] == 0:
					queue.append(neighbor_node)
		
		# 5. 순환 의존성 체크:
		#	처리된 노드의 수가 '관련된' 노드의 총 수와 다르면 순환 의존성이 존재합니다.
		if num_processed_nodes != len(all_relevant_nodes):
			raise ValueError("순환 의존성이 발견되었습니다. 업데이트 순서를 결정할 수 없습니다.")
		
		# 6. 최종 업데이트 순서 필터링:
		#	위상 정렬된 목록에서 'is_change'가 True인 노드들(실제로 업데이트가 필요한 노드)만 반환합니다.
		final_update_order = [node for node in topo_order_full if self.is_change[node]]

		return final_update_order

	# 사용되지 않음
	def has_circular_dependency(self) -> bool:
		"""
		전체 의존성 그래프에 순환 의존성이 존재하는지 여부를 검사합니다.
		이 메서드는 'is_change' 상태와 무관하게 모든 노드를 고려합니다.
		"""
		temp_graph = defaultdict(list)
		temp_indegree = defaultdict(int)
		all_nodes_in_graph = set()

		# 전체 그래프를 재구성하여 순환 의존성 검사에 사용
		for target, sources in self.dependencies.items():
			all_nodes_in_graph.add(target)
			for source in sources:
				all_nodes_in_graph.add(source)
				temp_graph[source].append(target) # source -> target 에지
				temp_indegree[target] += 1
		
		# 진입차수가 0인 노드들을 큐에 추가 (Kahn's Algorithm 시작)
		for node in all_nodes_in_graph:
			if node not in temp_indegree:
				temp_indegree[node] = 0

		queue = deque([node for node in all_nodes_in_graph if temp_indegree[node] == 0])
		count = 0 # 처리된 노드의 수

		while queue:
			node = queue.popleft()
			count += 1
			for neighbor in temp_graph[node]:
				temp_indegree[neighbor] -= 1
				if temp_indegree[neighbor] == 0:
					queue.append(neighbor)
		
		# 처리된 노드의 수가 전체 노드의 수와 다르면 순환 의존성이 존재합니다.
		return count != len(all_nodes_in_graph)

class DependentStates:
	"""
	변수들의 의존성을 추적하고 변경 시 자동 업데이트를 수행하는 클래스입니다.
	속성 직접 접근(states.var_name)을 통해 변수 값에 접근 및 설정합니다.
	"""
	def __init__(self):
		object.__setattr__(self, '_variables', {}) 
		object.__setattr__(self, '_graph', DependencyGraph())
		object.__setattr__(self, '_current_computing_var', None)

	def __getattr__(self, name: str) -> Any:
		"""
		states.변수명 형태로 속성에 접근할 때 호출됩니다.
		변수의 값을 반환하며, 계산 중인 경우 의존성을 추적합니다.
		"""
		if name in self._variables:
			# 현재 다른 변수의 compute_func가 실행 중이라면, 의존성 관계를 기록
			if self._current_computing_var:
				target = self._current_computing_var
				source = name
				# print(f"DEBUG: {target} 의존성 추적: {source}") # 디버깅용
				self._graph.add_dependency(target, source)
				# add_dependency 내부에서 _mark_as_changed(target)를 호출하므로,
				# 여기서 target의 is_change를 True로 설정할 필요는 없습니다.
			return self._variables[name].value
		
		# _variables에 없는 속성이면 기본 동작을 따름 (AttributeError 발생 또는 상위 클래스 탐색)
		# 중요: _graph, _current_computing_var 등의 내부 속성은 __init__에서 object.__setattr__로
		# 설정했으므로 여기에 걸리지 않습니다.
		raise AttributeError(f"'{type(self).__name__}' 객체는 '{name}' 속성을 가지고 있지 않습니다.")

	def __setattr__(self, name: str, value: Any) -> None:
		if name.startswith('_'): # 내부 속성 예외처리
			object.__setattr__(self, name, value)
		elif name in self._variables:
			var_obj = self._variables[name]
			if callable(value):
				var_obj.value = None
				var_obj.compute_func = value
				self._compute_and_set_value(name)
				# 기존에 있던 의존성은 _compute_and_set_value에서 초기화됨
			else:
				var_obj.value = value
				if var_obj.compute_func: # 계산 변수 -> 일반 변수
					self._graph.clean_dependency(name)
					var_obj.compute_func = None
				self._graph.set_changed(name)
		else: # 새로운 속성
			if callable(value):
				self._variables[name] = Variable(value=None, compute_func=value)
				self._compute_and_set_value(name)
			else:
				self._variables[name] = Variable(value=value, compute_func=None)
				self._graph.is_change[name] = False

	def _compute_and_set_value(self, name: str) -> None:
		"""
		지정된 변수의 compute_func를 실행하여 값을 계산하고 설정합니다.
		이 과정에서 의존성을 추적하고 기록합니다.
		"""
		variable = self._variables[name]
		if not variable.compute_func:
			return # 계산 함수가 없는 변수는 계산하지 않음

		# 이전에 설정된 의존성을 정리하여, 현재 계산에 필요한 의존성만 새로 추적
		# 이는 compute_func가 변경되거나, 내부 로직이 변경되어 의존성이 바뀔 수 있기 때문
		self._graph.clean_dependency(name)
		
		# 현재 계산 중인 변수를 설정 (의존성 추적 활성화)
		previous_computing_var = self._current_computing_var
		# _current_computing_var는 __setattr__에 걸리지 않도록 object.__setattr__ 사용
		object.__setattr__(self, '_current_computing_var', name)

		try:
			# compute_func 실행 시, 내부에서 self.변수명 접근을 통해 의존성이 자동으로 기록됨
			new_value = variable.compute_func(self)
			variable.value = new_value # Variable 객체의 value를 업데이트
			# 계산 완료 후 해당 변수는 더 이상 변경될 필요가 없으므로 is_change 상태 초기화
			self._graph.reset_change_status(name)
		finally:
			# 계산 완료 후 현재 계산 중인 변수 상태를 원래대로 복원
			object.__setattr__(self, '_current_computing_var', previous_computing_var)

	def update_all(self) -> None:
		"""
		현재 변경이 필요한 모든 변수들을 의존성 순서에 따라 업데이트합니다.
		"""
		try:
			update_order = self._graph.get_update_order()
			# print(f"DEBUG: Calculated update order: {update_order}") # 디버깅용

			for var_name in update_order:
				variable = self._variables[var_name]
				if variable.compute_func:
					# print(f"DEBUG: Updating '{var_name}'...") # 디버깅용
					self._compute_and_set_value(var_name)
				else:
					# compute_func가 없는 변수도 is_change가 True일 수 있음 (직접 set_value로 변경 등)
					# 이 경우, 값을 다시 계산할 필요는 없지만 is_change 상태는 리셋해야 함.
					self._graph.reset_change_status(var_name)
					# print(f"DEBUG: '{var_name}' has no compute_func, just resetting change status.")

		except ValueError as e:
			print(f"오류: 업데이트 중 순환 의존성 발생: {e}")
		except KeyError as e:
			print(f"오류: 업데이트 중 존재하지 않는 변수 참조: {e}")
		except Exception as e:
			print(f"알 수 없는 오류 발생: {e}")

	def get_all_var_names(self) -> List[str]:
		"""관리하는 모든 변수의 이름을 반환합니다."""
		return list(self._variables.keys())

	# 디버깅 또는 특정 경우에만 사용하도록 비공개화하거나 제거 고려
	# def get_variable_object(self, name: str) -> Variable:
	#	 """특정 변수 객체(Variable)를 반환합니다."""
	#	 if name not in self._variables:
	#		 raise KeyError(f"변수 '{name}'를 찾을 수 없습니다.")
	#	 return self._variables[name]

	def has_circular_dependency(self) -> bool:
		"""현재 의존성 그래프에 순환 의존성이 있는지 여부를 확인합니다."""
		return self._graph.has_circular_dependency()


if __name__ == "__main__":
	states = DependentStates()

	# 1. 변수 추가
	# 일반 변수: 초기값 설정
	states.input_a = 10
	states.input_b = 5
	states.constant_val = 20

	# 계산 변수: compute_func 지정 (이 방법으로만 등록 가능)
	# add_result: input_a 와 constant_val 에 의존
	# 이제 s.input_a 처럼 직접 속성 접근으로 값을 가져옵니다.
	def add_func(s: DependentStates):
		print(f"  [계산] add_result = input_a ({s.input_a}) + constant_val ({s.constant_val})")
		return s.input_a + s.constant_val

	# multiply_result: input_b 와 add_result 에 의존
	def multiply_func(s: DependentStates):
		print(f"  [계산] multiply_result = input_b ({s.input_b}) * add_result ({s.add_result})")
		return s.input_b * s.add_result

	# final_result: multiply_result 에 의존
	def final_func(s: DependentStates):
		print(f"  [계산] final_result = multiply_result ({s.multiply_result}) + 100")
		return s.multiply_result + 100

	states.add_result = add_func
	states.multiply_result = multiply_func
	states.final_result = final_func

	print("--- 초기 상태 및 값 ---")
	for var_name in states.get_all_var_names():
		# 이제 states.변수명 으로 직접 접근
		print(f"{var_name}: {getattr(states, var_name)} (is_change: {states._graph.is_change[var_name]})")

	print("\n--- 1 번째 업데이트 (초기 계산 및 의존성 설정) ---")
	states.update_all() 

	print("\n--- 업데이트 후 값 ---")
	for var_name in states.get_all_var_names():
		print(f"{var_name}: {getattr(states, var_name)} (is_change: {states._graph.is_change[var_name]})")
	
	# 일반 변수 -> 일반 변수
	print("\n--- input_a 값 변경 (자동 업데이트 트리거) ---")
	states.input_a = 15 # 직접 속성 접근으로 값 변경

	print("\n--- 값 변경 후 is_change 상태 확인 ---")
	for var_name in states.get_all_var_names():
		print(f"{var_name}: {states._graph.is_change[var_name]}")

	print("\n--- 2 번째 업데이트 실행 ---")
	states.update_all() # 변경된 변수와 종속 변수들만 업데이트됨

	print("\n--- 업데이트 후 최종 값 ---")
	for var_name in states.get_all_var_names():
		print(f"{var_name}: {getattr(states, var_name)} (is_change: {states._graph.is_change[var_name]})")

	# 일반 변수 -> 일반 변수
	print("\n--- input_b 값 변경 (자동 업데이트 트리거) ---")
	states.input_b = 10 # 직접 속성 접근으로 값 변경

	print("\n--- 값 변경 후 is_change 상태 확인 ---")
	for var_name in states.get_all_var_names():
		print(f"{var_name}: {states._graph.is_change[var_name]}")

	print("\n--- 3 번째 업데이트 실행 ---")
	states.update_all()

	print("\n--- 업데이트 후 최종 값 ---")
	for var_name in states.get_all_var_names():
		print(f"{var_name}: {getattr(states, var_name)} (is_change: {states._graph.is_change[var_name]})")

	# 계산 변수 -> 계산 변수
	print("\n--- multiply_result 값 변경 (자동 업데이트 트리거) ---")
	states.multiply_result = lambda s: s.input_a * s.input_b

	print("\n--- 값 변경 후 is_change 상태 확인 ---")
	for var_name in states.get_all_var_names():
		print(f"{var_name}: {states._graph.is_change[var_name]}")

	print("\n--- 4 번째 업데이트 실행 ---")
	states.update_all()

	print("\n--- 업데이트 후 최종 값 ---")
	for var_name in states.get_all_var_names():
		print(f"{var_name}: {getattr(states, var_name)} (is_change: {states._graph.is_change[var_name]})")

	# 일반 변수 -> 계산 변수
	print("\n--- input_b 값 변경 (자동 업데이트 트리거) ---")
	states.input_b = lambda s: s.input_a ** 2

	print("\n--- 값 변경 후 is_change 상태 확인 ---")
	for var_name in states.get_all_var_names():
		print(f"{var_name}: {states._graph.is_change[var_name]}")

	print("\n--- 5 번째 업데이트 실행 ---")
	states.update_all()

	print("\n--- 업데이트 후 최종 값 ---")
	for var_name in states.get_all_var_names():
		print(f"{var_name}: {getattr(states, var_name)} (is_change: {states._graph.is_change[var_name]})")

	# 계산 변수 -> 일반 변수
	print("\n--- input_b 값 변경 (자동 업데이트 트리거) ---")
	states.input_b = 30

	print("\n--- 값 변경 후 is_change 상태 확인 ---")
	for var_name in states.get_all_var_names():
		print(f"{var_name}: {states._graph.is_change[var_name]}")

	print("\n--- 6 번째 업데이트 실행 ---")
	states.update_all()

	print("\n--- 업데이트 후 최종 값 ---")
	for var_name in states.get_all_var_names():
		print(f"{var_name}: {getattr(states, var_name)} (is_change: {states._graph.is_change[var_name]})")

	print("\n--- 순환 의존성 테스트 ---")
	try:
		states.var_e = 1
		states.var_f = lambda s: s.var_e + 1
		
		# var_e -> var_f 의존성은 f의 compute_func에 의해 자동으로 추가되었음
		# 이제 var_f -> var_e 의존성을 수동으로 추가하여 순환 생성 (DependencyGraph에 직접 접근)
		states._graph.add_dependency("var_e", "var_f") # E <- F (F는 이미 E에 의존) -> E <-> F
		
		print(f"순환 의존성 존재 여부: {states.has_circular_dependency()}")
		states.var_e = 100 # 값 변경 트리거
		states.update_all() # 순환 의존성으로 인해 오류 발생 예상
	except ValueError as e:
		print(f"순환 의존성으로 인해 업데이트 실패: {e}")

	print("\n--- 순환 의존성 제거 후 테스트 ---")
	states._graph.remove_dependency("var_e", "var_f") # 순환 해제
	states.var_e = 200 # 값 재변경하여 업데이트 트리거
	print(f"순환 의존성 존재 여부: {states.has_circular_dependency()}")
	states.update_all() # 이제는 정상 업데이트 예상

	print(f"var_e: {states.var_e}")
	print(f"var_f: {states.var_f}")