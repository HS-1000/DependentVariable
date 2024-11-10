import dependent_variable as dv

states = dv.DependentStates()

# 초기값 설정
states.c = 5
print("Initial attributes:", states._attrs)

# 의존성 설정
states.b = lambda s: s.c * 2  # b = 10
states.a = lambda s: s.b * s.b  # a = 100
states.d = lambda s: s.c - s.a  # d = -95
states.e = lambda s: abs(s.c)  # e = 5

def f_update(s):
    return s.b if s.c > 0 else s.d

states.f = f_update  # f = 10

print("Dependency graph:\n", states.dependencies)

# 값 업데이트 시도
result = states.b = 123  # can't set value
if result is False:
    print("Failed to update states.b")

# 현재 값 출력
print("Current values:")
print("c:", states.c)  # 5
print("b:", states.b)  # 10
print("a:", states.a)  # 100
print("d:", states.d)  # -95
print("e:", states.e)  # 5
print("f:", states.f)  # 10

print("Setting value-----------------------------------------------------")
states.c = -5  # set origin value -> auto update
print("Updated values after changing c to -5:")
print("c:", states.c)  # -5
print("b:", states.b)  # -10
print("a:", states.a)  # -10
print("d:", states.d)  # 5
print("e:", states.e)  # 5
print("f:", states.f)  # d

print("Setting update function-------------------------------------------")
states.a = lambda s: s.c * 2
print("Updated values after changing a:")
print("c:", states.c)  # -5
print("b:", states.b)  # -10
print("a:", states.a)  # -10
print("d:", states.d)  # 5
print("e:", states.e)  # 5
print("f:", states.f)  # d

print("Deleting values---------------------------------------------------")
del states.c # False
del states.f # True

print("List index slice--------------------------------------------------")
g = [i for i in range(1, 10)]
states.g = g
del g
print(states.g[:5])

print("To dict-----------------------------------------------------------")
print(states.to_dict())

print("Use independent---------------------------------------------------")
states.independent['before_a'] = states.a
states.c = 25
print(states.to_dict())
print('Before a: ', states.independent['before_a'])


