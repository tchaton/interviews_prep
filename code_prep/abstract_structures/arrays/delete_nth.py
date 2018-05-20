"""
Given a list lst and a number N, create a new list
that contains each number of the list at most N times without reordering.
For example if N = 2, and the input is [1,2,3,1,2,1,2,3], you take [1,2,3,1,2],
drop the next [1,2] since this would lead to 1 and 2 being in the result 3 times, and then take 3,
which leads to [1,2,3,1,2,3]
"""


def has_key(d, key):
	try:
		b = d[key]
		return True
	except:
		return False

def create_element(d, key):
	if has_key(d, key):
		d[key] += 1
	else:
		d[key] = 0

def check_d(d, key, N):
	if d[key] < N:
		return True
	else:
		return False

def delete_nth(list, N):
	d = {}
	out = []
	for i in list:
		key = str(i)
		create_element(d, key)
		if check_d(d, key, N):
			out.append(i)
	return out

def test():
	arrs = ( [1,2,3,1,2,1,2,3], [1, 2, 3, 1, 2, 3])
	assert delete_nth(arrs[0], 2) == arrs[1], 'error'

	arrs = ( [1, 1, 1, 1, 1, 1, 5, 5, 5, 4 , 4 , 4 ], [1, 1, 1, 1, 5, 5, 5, 4, 4, 4])
	assert delete_nth(arrs[0], 4) == arrs[1], 'error'

	arrs = ([], [])
	assert delete_nth(arrs[0], 2) == arrs[1], 'error'
