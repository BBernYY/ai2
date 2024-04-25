from scipy.special import expit
import numpy as np
def calculate_model(sizes):
    global n
    global w
    global b
    global n_grad
    global w_grad
    global b_grad
    n = [np.zeros(sizes[i]) for i in range(len(sizes))]
    n_grad = [np.zeros(sizes[i]) for i in range(len(sizes))]

    w = [[(np.random.normal(1, 2, size=sizes[i-1]) if i != 0 else np.zeros(sizes[i])) for _ in range(sizes[i])] for i in range(len(sizes))]
    w_grad = [[np.zeros(sizes[i-1]) for _ in range(sizes[i])] for i in range(len(sizes))]

    b = [[(np.random.uniform(-5, 5) if i != 0 else 0) for _ in range(sizes[i])]  for i in range(len(sizes))]
    b_grad = [[0 for _ in range(sizes[i])]  for i in range(len(sizes))]
def deriv_sigmoid(x):
    return expit(x)*(1-expit(x))
def z(h, v):
    return np.dot(w[h][v], n[h-1])+b[h][v]

def calculate():
    global n
    global w
    global b
    for i in range(1, len(n)):
        for j in range(len(n[i])):
            n[i][j] = expit(z(i, j))

# def prop_bias(n):
#     calculate()
#     return deriv_sigmoid(z(n))*2*(nodes[n]-expected_value)
# def prop_weight(n):
#     calculate()
#     return nodes[n-1]*deriv_sigmoid(z(n))*2*(nodes[n]-expected_value)

# def prop(inde, typ):
#     calculate()
#     t = len(nodes)-1
#     num = 2*(nodes[t]-expected_value)
#     while True:
#         num *= deriv_sigmoid(z(t))
#         if t == inde:
#             if typ == 'w':
#                 num *= nodes[t-1]
#             break
#         num *= weights[t]
#         t -= 1
#     return num

def total_cost(ev):
    tot = sum([(n[-1][i]-ev[i])**2 for i in range(sizes[-1])])
    return tot
    

# def prop_all():
#     calculate()
#     gradient = {}
#     def rec(typ, h, v, v2, funcs):
#         match typ:
#             case 'w':
#                 w_grad[h][v][v2] = num
#             case 'b':
#                 b_grad[h][v] = num
#             case 'n':
#                 funcs.append(lambda v: deriv_sigmoid(z(h, v)))
#                 for i in range(len(w[h][v])):
                    

#                 rec(num*nodes[t-1], 'w', t)
#                 rec(num, 'b', t)
#                 if t > 1:
#                     rec(num*weights[t], 'n', t-1)
#     h, v = len(sizes)-1, 0
#     lambda v: 2*(n[h][v]-expected_values[v])
#     rec(, 'n', t)
#     return gradient
def prop2(h, v):
    global w_grad
    global b_grad
    for i in range(len(w[h][v])):
        w_grad[h][v][i] += n_grad[h][v]*deriv_sigmoid(z(h, v))*n[h-1][i]
    b_grad[h][v] += n_grad[h][v]*deriv_sigmoid(z(h, v))


def prop3(h, v):
    for i in range(len(n[h+1])):
        n_grad[h][v] += n_grad[h+1][i]*deriv_sigmoid(z(h, v))*w[h+1][i][v]

def adjust(ev):
    global w_grad
    global b_grad
    global w
    global b
    for h in range(len(n)):
        for v in range(len(n[h])):
            tot = total_cost(ev)
            w[h][v] += -w_grad[h][v]*STEPSIZE
            b[h][v] += -b_grad[h][v]*STEPSIZE


def full_prop(ev):
    calculate()
    global n_grad
    global w
    global b
    for h in reversed(range(len(sizes))):
        for v in range(len(n[h])):
            if h == len(sizes)-1:
                n_grad[h][v] = 2*(n[h][v] - ev[v])
            else:
                prop3(h, v)
            prop2(h, v)
        
        
    
    

sizes = [2, 3, 3, 2]
calculate_model(sizes)
STEPSIZE = 10
inputs = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
]
outputs = [
    [1.0, 1.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 0.0]
]
def main():
    global n
    t = 0
    cost_sum = 0
    while True:
        if np.random.rand() > 0.5:
            i = np.random.randint(0,4)
            ev = outputs[i]
            n[0] = np.array(inputs[i])
        else:
            ev = [1.0, 0.0]
            n[0] = np.array([0.0,1.0])
        full_prop(ev)

        t += 1
        cost_sum += total_cost(ev)
        if np.random.rand() > 0.999:
            print(str(cost_sum/t)+"\t"+str(total_cost(ev)))
            adjust(ev)
            
            
main()