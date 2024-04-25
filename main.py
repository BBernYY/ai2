from scipy.special import expit

nodes = [0, 0, 0]
weights = [0, 2, 3]
biases = [0, 0.2, -3]
expected_value = 1
def deriv_sigmoid(x):
    return expit(x)*(1-expit(x))
def calculate():
    global nodes
    global weights
    for i in range(1, len(nodes)):
        nodes[i] = expit(z(i))

def z(n):
    return nodes[n-1]*weights[n]+biases[n]

def prop_bias(n):
    calculate()
    return deriv_sigmoid(z(n))*2*(nodes[n]-expected_value)
def prop_weight(n):
    calculate()
    return nodes[n-1]*deriv_sigmoid(z(n))*2*(nodes[n]-expected_value)

def prop(inde, typ):
    calculate()
    t = len(nodes)-1
    num = 2*(nodes[t]-expected_value)
    while True:
        num *= deriv_sigmoid(z(t))
        if t == inde:
            if typ == 'w':
                num *= nodes[t-1]
            break
        num *= weights[t]
        t -= 1
    return num
    
def prop_all():
    calculate()
    gradient = {}
    def rec(num, typ, t):
        if typ != 'n':
            gradient[typ+str(t)] = num
        else:
            num *= deriv_sigmoid(z(t))
            rec(num*nodes[t-1], 'w', t)
            rec(num, 'b', t)
            if t > 1:
                rec(num*weights[t], 'n', t-1)
    t = len(nodes)-1
    rec(2*(nodes[t]-expected_value), 'n', t)
    return gradient

print(prop_all())